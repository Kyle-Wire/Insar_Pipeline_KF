"""
Validation harness (PLAN phase 1, V1): the pipeline's InSAR against
independent ground truth, under a protocol fixed in writing before any result
was seen (review_2026-10-08/protocol_*.yaml).

    python -m insar_pipeline.cli validate <config.yaml> --protocol <protocol.yaml> [--out DIR]
        [--run ID] [--cache DIR] [--ga-xlsx FILE] [--points FILE] [--up FILE]
        [--outlines FILE] [--lines FILE] [--no-sensitivity] [--no-epoch-coverage]

It READS only: the run's delivered MintPy products (the time series the
velocity was fitted to - deliverables.timeseries_file - its temporal
coherence and per-pixel geometry), the pipeline's stated rate uncertainty
(deliverables/uncertainty_model.json, else MintPy's formal velocityStd) and
the ground truth through ground_truth.py (NGL IGS20 tenv3 with steps.txt, the
Geoscience Australia CEMP xlsx, user CSV points) and, when given, D1's
decomposed Up series. It WRITES validation.json, validation.md,
validation_epochs.csv and a few plots into --out (default
<project>/validation/protocol_<id>_<run>) and nothing else: no pipeline
product and no client number moves.

How a comparison is made (protocol sections in brackets):
  - InSAR at a point and date = the median of the reliable cells (value
    finite, temporal coherence >= the floor) whose centres lie within the
    radius; fewer than the minimum cells or independent radar pixels = no
    value at that date, never filled [sampling, reliability]. Distances are
    geodesic on a lat/lon grid and Euclidean on a projected grid.
  - GNSS at a SAR date = the mean of the daily solutions within +-N days
    [epoch_matching]; daily outliers beyond 5 robust sigma about the model
    are dropped; the per-epoch sigma is the robust sigma of the daily
    residuals, not divided by sqrt(n) [gnss_screening].
  - GNSS east/north/up -> LOS with the per-pixel LOS unit vector (ground to
    satellite, MintPy convention: uplift positive), the median over the same
    cells.
  - The datum is never fitted to a scored station:
      reference-station design (protocol A): double difference to the
      reference station, l(P) . {dENU(P) - dENU(ref)}: plate motion and the
      GNSS frame cancel in the station difference, the InSAR datum in the
      InSAR difference;
      network design (protocols B1, B2): station-pair double differences
      (each station's GNSS projected with its own LOS vector), and
      GNSS-absolute leave-one-out (one offset per epoch plus one constant per
      calibration station, fitted without the scored station).
  - Rates: the protocol's model (piecewise linear by water year, or linear;
    plus an annual term) fitted to both series on the matched epochs only.
  - Statistics carry bootstrap 95% intervals over stations (10000 resamples,
    seed 20261008, both from the protocol). Each graded row is read against
    the protocol's own threshold text: pass / usable with stated caveats /
    fail. A row the harness cannot compute yet says why; it is never passed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import logging
import math
import re
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger(__name__)

PKG = Path(__file__).resolve().parent
README_HASHES = PKG / "configs" / "experiments" / "README.md"
# PLAN T1's digitised panel outlines: where amendment 2026-10-08/1 of protocol A places its stable ring
T1_OUTLINES = PKG / "review_2026-10-08" / "truth" / "panels.geojson"
# ... and its digitised Tahmoor West GNSS units (ground_truth.load_points_csv; tahmoor_w_units)
T1_POINTS = PKG / "review_2026-10-08" / "truth" / "tahmoor_west_gnss_units.csv"
DEFAULT_SEED = 20261008
DEFAULT_BOOT = 10000
DEFAULT_MATCH_DAYS = 3
SCREEN_RUNNING_MEDIAN_DAYS = 31     # the screening model where the protocol has no rate model (events)
LINEAR_MIN_YEARS = 1.0              # a linear rate needs this much matched record
LINEAR_MIN_EPOCHS = 10
USABLE_SHARE = 0.80                 # protocol A: reference usable at >= 80% of compared epochs

GRADE_TEXT = {"pass": "pass", "marginal": "usable with stated caveats", "fail": "fail",
              "not_testable": "not testable", "not_computed": "not computed",
              "report_only": "reported, not graded", "not_reached": "no verdict yet"}
DATE_RE = r"\d{4}-\d{2}-\d{2}"
ALIAS_TIF = "quality_aliasing_risk.tif"      # C-alias (F2): 0 no risk found, 1 risk, 2 likely aliased; 255 outside
ALIAS_JSON = "aliasing_risk.json"            # C-alias: method, thresholds, per-pair and per-interval summary
ALIAS_FLAGGED = (1, 2)                       # the levels that flag (a lower bound: 0 is no evidence, not proof)


# =============================================================== small helpers

def _s(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _day(d) -> int:
    """'YYYYMMDD', 'YYYY-MM-DD', a date or an int -> days since 1970-01-01."""
    if isinstance(d, (int, np.integer)):
        return int(d)
    if isinstance(d, np.datetime64):
        return int(d.astype("datetime64[D]").astype(np.int64))
    s = str(d).strip()[:10].replace("-", "")
    return int(np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:8]}", "D").astype(np.int64))


def _days(ds) -> np.ndarray:
    return np.array([_day(d) for d in ds], dtype=np.int64)


def _iso(day) -> str:
    return str(np.datetime64(int(day), "D"))


def _f(x, nd=3):
    """JSON-safe rounded float (None for missing)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _num(pattern, text, default, cast=float):
    m = re.search(pattern, str(text or ""), re.I)
    return cast(m.group(1)) if m else default


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _f(o, 6)
    if isinstance(o, np.ndarray):
        return [_json_default(v) if isinstance(v, (np.generic, np.ndarray)) else v for v in o.tolist()]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# =============================================================== protocol

def fixed_hash_status(path) -> dict:
    """Whether the protocol file is one of those fixed on 8 Oct 2026, unchanged
    (configs/experiments/README.md holds their sha256)."""
    path = Path(path)
    h = _sha256(path)
    try:
        text = README_HASHES.read_text()
    except OSError:
        text = ""
    m = re.search(r"`" + re.escape(path.name) + r"`\s*\|\s*`([0-9a-f]{64})`", text)
    if not m:
        return {"status": "ad_hoc", "sha256": h,
                "text": "not one of the protocols fixed in configs/experiments/README.md: an ad hoc "
                        "protocol, so these are not experiment results"}
    if m.group(1) == h:
        return {"status": "fixed", "sha256": h,
                "text": "matches the sha256 fixed in configs/experiments/README.md"}
    return {"status": "changed", "sha256": h, "fixed_sha256": m.group(1),
            "text": f"does NOT match the sha256 fixed in configs/experiments/README.md "
                    f"({m.group(1)[:12]}...): the file was edited after it was fixed"}


def _merge(base, over):
    if over is None:
        return copy.deepcopy(base)
    if isinstance(base, dict) and isinstance(over, dict):
        out = copy.deepcopy(base)
        for k, v in over.items():
            out[k] = _merge(base.get(k), v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(over)


def load_protocol(path) -> dict:
    """The protocol YAML with its sha256 and, for `inherits:`, the named
    sections of the parent protocol merged under the child's own keys."""
    path = Path(path).resolve()
    P = yaml.safe_load(path.read_text())
    if not isinstance(P, dict) or "protocol" not in P:
        raise ValueError(f"{path} is not a validation protocol (no 'protocol:' key)")
    P["_path"], P["_sha256"] = str(path), _sha256(path)
    inh = P.get("inherits")
    if isinstance(inh, dict) and inh.get("file"):
        cands = [PKG / inh["file"], path.parent / Path(inh["file"]).name, PKG.parent / inh["file"]]
        parent_path = next((c for c in cands if c.exists()), None)
        if parent_path is None:
            raise FileNotFoundError(f"{path.name} inherits from {inh['file']}, which was not found")
        parent = load_protocol(parent_path)
        if inh.get("version") is not None and parent.get("version") != inh.get("version"):
            raise ValueError(f"{path.name} inherits version {inh.get('version')} of {parent_path.name}, "
                             f"which is now version {parent.get('version')}")
        for sec in inh.get("sections", []):
            if sec in parent:
                P[sec] = _merge(parent[sec], P.get(sec))
        P["_inherited"] = {"file": str(parent_path), "sha256": parent["_sha256"],
                           "sections": list(inh.get("sections", []))}
    return P


# =============================================================== amendments

def amendment_switches(P) -> dict:
    """The machine-readable `harness:` keys of the protocol's dated
    amendments, merged in file order (a later amendment wins), with the ids
    that set them. Empty for a protocol without amendments: the harness then
    reads the protocol exactly as first fixed."""
    out, by = {}, {}
    for a in (P.get("amendments") or []) if isinstance(P, dict) else []:
        if not isinstance(a, dict):
            continue
        for k, v in (a.get("harness") or {}).items():
            if k == "run_configs" and isinstance(v, dict):
                # one config per run, or a list where a run is several projects (one geometry each)
                out.setdefault("run_configs", {}).update(
                    {str(r): ([str(x) for x in c] if isinstance(c, (list, tuple)) else str(c)) for r, c in v.items()})
            else:
                out[str(k)] = v
            by[str(k)] = str(a.get("id", "?"))
    if out:
        out["_by"] = by
    return out


def version_1(P) -> dict:
    """The protocol as first fixed: the same dict without its amendments."""
    Q = {k: copy.deepcopy(v) for k, v in P.items() if k != "amendments"}
    Q["amendments"] = []
    Q["_version_1_of"] = P.get("_path")
    return Q


def fixed_text_status(path) -> dict:
    """Whether the text above the protocol's `amendments:` key is the text
    fixed on 8 Oct 2026, byte for byte: the file cut at its last top-level
    `amendments:` line, plus `amendments: []` (how it ended when it was
    fixed), against the 'as first fixed' sha256 in
    configs/experiments/README.md. This is what shows that the amendments were
    appended and nothing fixed was rewritten."""
    path = Path(path)
    data = path.read_bytes()
    i = data.rfind(b"\namendments:")
    if i < 0:
        return {"status": "no_amendments_key", "text": "the file has no top-level 'amendments:' key"}
    h = hashlib.sha256(data[:i + 1] + b"amendments: []\n").hexdigest()
    try:
        text = README_HASHES.read_text()
    except OSError:
        text = ""
    m = re.search(re.escape(path.name) + r" \(first fixed\)\s*\|\s*`([0-9a-f]{64})`", text)
    if not m:
        return {"status": "ad_hoc", "sha256": h,
                "text": "no 'as first fixed' sha256 for this file name in configs/experiments/README.md"}
    if m.group(1) == h:
        return {"status": "unchanged", "sha256": h,
                "text": "the text above 'amendments:' is the text first fixed on 8 Oct 2026, byte for byte"}
    return {"status": "changed", "sha256": h, "fixed_sha256": m.group(1),
            "text": "the text above 'amendments:' differs from the text first fixed on 8 Oct 2026: fixed text "
                    "was rewritten, not amended"}


def _roles(entry) -> list[str]:
    r = entry.get("roles", entry.get("role", []))
    return [str(x) for x in (r if isinstance(r, list) else [r])]


def _point_role(roles) -> str | None:
    """The ground_truth Point role for a station's protocol roles: the one
    mapping ground_truth.protocol_points uses (reference -> reference;
    validation -> validation; calibration_pool -> calibration; null_test and
    reference_check -> null, because an alternative datum station is scored
    against zero motion whenever it is not the datum; not_used and
    duplicate_check -> None, not scored). Recorded in validation.json only;
    which station plays which part is read from the protocol roles."""
    gt = importlib.import_module(f"{__package__}.ground_truth" if __package__ else "ground_truth")
    return gt.protocol_role(roles)


def _stations_named(text, ids) -> list[str]:
    """Station ids in the order they appear in a prose field."""
    found = []
    for m in re.finditer(r"\b([A-Z0-9]{4})\b", str(text or "")):
        if m.group(1) in ids and m.group(1) not in found:
            found.append(m.group(1))
    return found


def protocol_params(P, run=None, backend=None) -> dict:
    """The protocol's machine-readable choices. Prose fields are read with
    narrow patterns; whatever is read is written into validation.json so a
    reader can check it against the protocol text."""
    samp, rel = P.get("sampling", {}) or {}, P.get("reliability", {}) or {}
    nisar = backend == "nisar"
    radius = float(samp.get("radius_m_nisar") if nisar and samp.get("radius_m_nisar") else samp.get("radius_m", 100))
    minimum = samp.get("minimum", "")
    min_cells = _num(r"(\d+)\s+reliable posting cells", minimum, 5, int)
    if nisar:
        min_cells = _num(r"NISAR:\s*(\d+)\s*cells", minimum, min_cells, int)
    min_indep = _num(r"(\d+)\s+independent radar pixels", minimum, 2, int)
    floor = float(rel.get("coherence_floor", 0.4))
    floors = [float(x) for x in rel.get("sensitivity_floors", []) or []]
    em = P.get("epoch_matching", {}) or {}
    match_days = _num(r"\+-\s*(\d+)\s*d", em.get("gnss_at_sar_date"), DEFAULT_MATCH_DAYS, int)
    survey_gap = _num(r"at most\s+(\d+)\s+days apart", em.get("survey_epochs"), 24, int)

    win = P.get("window", {}) or {}
    runs = P.get("runs", {}) or {}
    rwin = (runs.get(run) or {}).get("window") if run else None
    start, end = (rwin if isinstance(rwin, list) and len(rwin) == 2 else (win.get("start"), win.get("end")))
    aoi = ((P.get("site") or {}).get("aoi") or {})
    stations = {}
    for sid, e in ((P.get("truth") or {}).get("stations") or {}).items():
        if not isinstance(e, dict) or "lat" not in e or "lon" not in e:
            continue
        roles = _roles(e)
        if roles == ["not_used"]:
            continue
        stations[str(sid)] = {"lat": float(e["lat"]), "lon": float(e["lon"]), "roles": roles,
                              "role": _point_role(roles)}

    ref = P.get("reference", {}) or {}
    prim = ref.get("primary") if isinstance(ref.get("primary"), dict) else None
    if prim and prim.get("station"):
        design = "reference_station"
    elif isinstance(ref.get("modes"), dict) and "double_difference" in ref["modes"]:
        design = "network"
    else:
        design = None

    models = P.get("models", {}) or {}
    rate_text = str(models.get("rate_model") or models.get("stable_ground_rate") or "")
    hinges = re.findall(DATE_RE, rate_text) if "piecewise" in rate_text.lower() else []
    compared = models.get("compared_rates")
    wy_text = compared.get("water_year") if isinstance(compared, dict) else None
    water_years = []
    if wy_text and hinges and start and end:
        bounds = [start] + hinges + [str(np.datetime64(end) + np.timedelta64(1, "D"))]
        for a, b in zip(bounds[:-1], bounds[1:]):
            label = f"WY{(np.datetime64(b) - np.timedelta64(1, 'D')).astype(object).year}"
            water_years.append((label, _day(a), _day(b)))
    stats = P.get("statistics_reported", {}) or {}
    ci = stats.get("confidence_intervals", "")
    screen = (P.get("gnss_screening") or {}).get("method", "")

    out = {
        "protocol": str(P.get("protocol")), "version": P.get("version"), "run": run, "design": design,
        "radius_m": radius, "min_cells": min_cells, "min_independent_pixels": min_indep,
        "floor": floor, "sensitivity_floors": floors, "match_days": match_days,
        "survey_max_gap_days": survey_gap,
        "window": [str(start) if start else None, str(end) if end else None],
        "bbox_wsen": list(aoi.get("bbox_wsen") or []),
        "stations": stations,
        "hinges": hinges, "hinge_days": [_day(h) for h in hinges],
        "annual": "annual" in rate_text.lower(),
        "model_text": rate_text.strip(),
        "water_years": water_years,
        "wy_min_years": _num(r">=\s*([\d.]+)\s*yr", wy_text, 0.75),
        "wy_min_epochs": _num(r">=\s*(\d+)\s*matched epochs", wy_text, 10, int),
        "n_boot": _num(r"(\d+)\s+resamples", ci, DEFAULT_BOOT, int),
        "seed": _num(r"seed\s+(\d+)", ci, DEFAULT_SEED, int),
        "screen_sigma": _num(r"beyond\s+([\d.]+)\s+robust sigma", screen, 5.0),
        # protocol A: "equipment/earthquake steps from steps.txt"; B1 (and B2 by inheritance): "steps from
        # steps.txt" - steps.txt lists both kinds, so both are removed as offsets
        "earthquake_steps": bool(re.search(r"earthquake|steps from steps\.txt", str(screen), re.I)),
        "screen_model": "rate model" if rate_text and design == "network" else "running median",
        "rate_stations": [s for s, e in stations.items() if "rate" in e["roles"]],
        "series_stations": [s for s, e in stations.items() if "series" in e["roles"]],
        "calibration_pool": [s for s, e in stations.items() if "calibration_pool" in e["roles"]],
    }
    sw = amendment_switches(P)
    out["amend"] = {k: v for k, v in sw.items() if k != "run_configs"}
    out["change_intervals"] = _change_intervals(sw, water_years, start, end)
    out["change_min_years"] = float(sw.get("change_min_years", out["wy_min_years"]))
    if design == "reference_station":
        out["reference_station"] = str(prim["station"])
        out["reference_fallback"] = [s for s in _stations_named(prim.get("fallback_if_unusable"), stations)
                                     if s != out["reference_station"]]
        out["usable_share"] = _num(r"fewer than\s+(\d+)%", prim.get("fallback_if_unusable"), 80, float) / 100.0
        order = []
        for item in prim.get("source_order") or []:
            src = "ga" if str(item).strip().upper().startswith("GA") else "ngl" if "NGL" in str(item).upper() else None
            if src and src not in order:
                order.append(src)
        out["source_order"] = order or ["ngl"]
        out["reference_sensitivity"] = [s for t in (ref.get("sensitivity") or [])
                                        for s in _stations_named(t, stations)]
        out["null_stations"] = [s for s, e in stations.items() if "null_test" in e["roles"]]
        out["validation_stations"] = [s for s, e in stations.items() if "validation" in e["roles"]]
    return out


def _change_intervals(sw, water_years, start, end) -> list:
    """The intervals the amended york_slope row takes per-station changes
    over (amendment 2026-10-08/2): the protocol's water years (B1), or one-year
    intervals from the window start (B2's 'annual changes'), as (label,
    first day, day after the last)."""
    how = sw.get("change_intervals")
    if not how or not start or not end:
        return []
    if how == "water_years":
        return list(water_years)
    if how == "annual_from_window_start":
        out, a = [], np.datetime64(str(start))
        stop = np.datetime64(str(end)) + np.timedelta64(1, "D")
        while a < stop:
            y, m, d = str(a).split("-")
            b = np.datetime64(f"{int(y) + 1:04d}-{m}-{d}")
            out.append((f"Y{str(a)}", _day(a), _day(min(b, stop))))
            a = b
        return out
    raise ValueError(f"amendment key change_intervals {how!r} is not understood")


# =============================================================== grades

class GradeError(ValueError):
    pass


_METRIC_WORDS = {"median rmse": "median_rmse", "rmse_vs_zero": "rmse_vs_zero", "rmse": "rmse",
                 "|bias|": "abs_bias", "share_beyond": "share_beyond", "ratio": "ratio", "rate": "rate", "r": "r"}


def _parse_part(part: str, default_share: str | None = None) -> dict:
    low = part.strip().rstrip(".").strip().lower()
    if re.fullmatch(r"95% ci includes 1", low):
        return {"metric": "ci_includes_1", "op": "is", "value": True}
    m = re.fullmatch(r"(?:slope )?within ([\d.]+)\s*-\s*([\d.]+)", low)
    if m:
        return {"metric": "slope", "op": "within", "value": [float(m[1]), float(m[2])]}
    m = re.fullmatch(r"\|err\|\s*<=\s*max\(\s*([\d.]+)\s*mm\s*,\s*([\d.]+)\s*x\s*\|truth\|\s*\)", low)
    if m:
        return {"metric": "abs_err", "op": "<=max", "value": [float(m[1]), float(m[2])]}
    m = re.fullmatch(r"\|err\|\s*<=\s*([\d.]+)\s*x\s*\|truth\|", low)
    if m:
        return {"metric": "abs_err", "op": "<=frac", "value": float(m[1])}
    m = re.fullmatch(r"\|err\|\s*<=\s*([\d.]+)\s*mm\s+at every station", low)
    if m:
        return {"metric": "max_abs_err", "op": "<=", "value": float(m[1])}
    m = re.fullmatch(r"\|err\|\s*<=\s*([\d.]+)\s*mm", low)
    if m:
        return {"metric": "abs_err", "op": "<=", "value": float(m[1])}
    m = re.fullmatch(r"([\d.]+)\s*<=\s*([a-z_]+)\s*<\s*([\d.]+)", low)
    if m:
        return {"metric": m[2], "op": "in[)", "value": [float(m[1]), float(m[3])]}
    m = re.fullmatch(r"(median rmse|rmse_vs_zero|rmse|\|bias\||share_beyond|ratio|rate|r)\s*(<=|>=|<|>)\s*"
                     r"([\d.]+)\s*(?:mm/yr|mm)?", low)
    if m:
        return {"metric": _METRIC_WORDS[m[1]], "op": m[2], "value": float(m[3])}
    m = re.fullmatch(r"(<=|>=|<|>)\s*([\d.]+)\s*%\s*(.*)", low)
    if m:
        rest = m[3].strip()
        if not rest:
            metric = default_share
        elif rest.startswith("inside"):
            metric = "share_inside"
        elif rest.startswith("of cells"):
            metric = "share_cells"
        elif "meet" in rest:
            metric = "share_meeting"
        else:
            metric = None
        if metric is None:
            raise GradeError(f"threshold text not understood: {part!r}")
        return {"metric": metric, "op": m[1], "value": float(m[2]) / 100.0}
    m = re.fullmatch(r"at most (\d+) outside", low)
    if m:
        return {"metric": "n_outside", "op": "<=", "value": int(m[1])}
    raise GradeError(f"threshold text not understood: {part!r}")


def parse_grade_text(text, default_share: str | None = None) -> list[list[dict]]:
    """One grade string -> alternatives (OR) of condition groups (AND).

    A remark in brackets after a space is commentary ("(the TRE bar)"),
    except a count rule "(6 stations: at most 1 outside)", which is the
    protocol's own reading of a share for its station count and so is an
    alternative: >= 90% inside OR (at least 6 scored AND at most 1 outside).
    The count holds only at the station count it was written for: with 3
    scored, 1 outside is 67% inside, which the share rule does not pass."""
    alts: list[list[dict]] = []

    def remark(m):
        cnt = re.search(r"at most \d+ outside", m.group(1))
        if cnt:
            grp = [_parse_part(cnt.group(0))]
            n_for = re.search(r"(\d+)\s+(?:stations|sites|points)\b", m.group(1))
            if n_for:
                grp.append({"metric": "n", "op": ">=", "value": int(n_for.group(1))})
            alts.append(grp)
        return ""

    main = re.sub(r"\s\(([^()]*)\)", remark, str(text)).strip()
    parts = [p for p in re.split(r"\s+AND\s+", main) if p.strip()]
    if not parts:
        raise GradeError(f"empty threshold text: {text!r}")
    return [[_parse_part(p, default_share) for p in parts]] + alts


def _cond(c: dict, M: dict):
    """True / False / None (metric missing), always a plain bool: a numpy
    bool is never `is False`, which once graded a failed row as a pass."""
    r = _cond_raw(c, M)
    return None if r is None else bool(r)


def _cond_raw(c: dict, M: dict):
    v = M.get(c["metric"])
    if v is None:
        return None
    if not isinstance(v, (bool, np.bool_)):
        v = float(v)
        if not math.isfinite(v):
            return None
    op, x = c["op"], c["value"]
    if op == "is":
        return bool(v) == x
    if op == "within":
        return x[0] <= v <= x[1]
    if op == "in[)":
        return x[0] <= v < x[1]
    if op in ("<=max", "<=frac"):
        t = M.get("truth")
        if t is None or not math.isfinite(float(t)):
            return None
        t = float(t)
        lim = max(x[0], x[1] * abs(t)) if op == "<=max" else x * abs(t)
        return v <= lim
    return {"<=": v <= x, ">=": v >= x, "<": v < x, ">": v > x}[op]


def _eval_groups(groups, M):
    vals = []
    for g in groups:
        r = [_cond(c, M) for c in g]
        vals.append(False if any(v is False for v in r) else None if any(v is None for v in r) else True)
    if any(v is True for v in vals):
        return True
    if any(v is None for v in vals):
        return None
    return False


def grade_row(spec, M: dict) -> dict:
    """Grade metrics M against a protocol row's `grade` (pass, then marginal,
    else fail). A metric the row needs but M lacks makes it 'not testable';
    a threshold the parser cannot read is 'not computed', never a pass."""
    if not isinstance(spec, dict) or "pass" not in spec:
        return {"grade": "report_only", "label": GRADE_TEXT["report_only"], "rule": spec}
    rule = {k: spec.get(k) for k in ("pass", "marginal", "fail") if k in spec}
    try:
        pg = parse_grade_text(spec["pass"])
        share = next((c["metric"] for g in pg for c in g if c["metric"].startswith("share")), None)
        mg = parse_grade_text(spec["marginal"], share) if spec.get("marginal") else []
    except GradeError as exc:
        return {"grade": "not_computed", "label": GRADE_TEXT["not_computed"], "rule": rule, "why": str(exc)}
    needed = sorted({c["metric"] for g in pg + mg for c in g})
    missing = [m for m in needed if M.get(m) is None or (isinstance(M.get(m), (float, np.floating))
                                                         and not math.isfinite(M[m]))]
    p = _eval_groups(pg, M)
    if p is True:
        g = "pass"
    elif p is None:
        g = "not_testable"
    else:
        q = _eval_groups(mg, M) if mg else False
        g = "marginal" if q is True else "not_testable" if q is None else "fail"
    out = {"grade": g, "label": GRADE_TEXT[g], "rule": rule}
    if g == "not_testable":
        out["missing"] = missing
    return out


def _labelled(grade: str, why: str = "", **kw) -> dict:
    d = {"grade": grade, "label": GRADE_TEXT[grade]}
    if why:
        d["why"] = why
    d.update(kw)
    return d


# =============================================================== statistics

def error_stats(err) -> dict:
    e = np.asarray(err, float)
    e = e[np.isfinite(e)]
    if not e.size:
        return {"n": 0}
    return {"n": int(e.size), "bias": float(e.mean()), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))), "median_abs": float(np.median(np.abs(e)))}


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def york_fit(x, y, sx, sy, max_iter: int = 200, tol: float = 1e-12) -> dict | None:
    """Straight line with errors in both x and y (York et al. 2004, no
    correlation between the errors): slope, intercept and their 1-sigma."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx, sy = np.asarray(sx, float) * np.ones_like(x), np.asarray(sy, float) * np.ones_like(y)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sx) & np.isfinite(sy)
    x, y, sx, sy = x[ok], y[ok], sx[ok], sy[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return None
    floor = 1e-9 * max(1.0, float(np.max(np.abs(np.r_[x, y]))))
    sx, sy = np.maximum(sx, floor), np.maximum(sy, floor)
    b = float(np.polyfit(x, y, 1)[0])
    for _ in range(max_iter):
        W = 1.0 / (sy ** 2 + b * b * sx ** 2)
        Xb, Yb = np.sum(W * x) / np.sum(W), np.sum(W * y) / np.sum(W)
        U, V = x - Xb, y - Yb
        beta = W * (U * sy ** 2 + b * V * sx ** 2)
        den = np.sum(W * beta * U)
        if den == 0:
            return None
        b_new = float(np.sum(W * beta * V) / den)
        done = abs(b_new - b) <= tol * max(1.0, abs(b))
        b = b_new
        if done:
            break
    W = 1.0 / (sy ** 2 + b * b * sx ** 2)
    Xb, Yb = np.sum(W * x) / np.sum(W), np.sum(W * y) / np.sum(W)
    U, V = x - Xb, y - Yb
    beta = W * (U * sy ** 2 + b * V * sx ** 2)
    a = Yb - b * Xb
    xa = Xb + beta
    xm = np.sum(W * xa) / np.sum(W)
    u = xa - xm
    sb = math.sqrt(1.0 / np.sum(W * u * u)) if np.sum(W * u * u) > 0 else float("nan")
    sa = math.sqrt(1.0 / np.sum(W) + xm ** 2 * sb ** 2) if math.isfinite(sb) else float("nan")
    mswd = float(np.sum(W * (y - b * x - a) ** 2) / (x.size - 2)) if x.size > 2 else float("nan")
    return {"slope": b, "intercept": float(a), "slope_se": sb, "intercept_se": sa, "mswd": mswd, "n": int(x.size)}


def bootstrap_ci(n_groups: int, stat, n_boot: int = DEFAULT_BOOT, seed: int = DEFAULT_SEED) -> dict:
    """Percentile 95% interval of stat(idx, rng) over resamples of the groups
    (stations or truth points) with replacement. Resamples where the
    statistic cannot be formed are left out and counted."""
    if n_groups < 2:
        return {"lo": None, "hi": None, "n_ok": 0, "resamples": 0,
                "why": "fewer than 2 stations or points: no interval can be formed"}
    rng = np.random.default_rng(seed)
    vals = np.full(n_boot, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in range(n_boot):
            idx = rng.integers(0, n_groups, n_groups)
            try:
                v = stat(idx, rng)
            except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError, TypeError):
                v = np.nan
            vals[b] = np.nan if v is None else v
    ok = vals[np.isfinite(vals)]
    out = {"resamples": int(n_boot), "seed": int(seed), "n_ok": int(ok.size)}
    if ok.size < max(10, 0.5 * n_boot):
        out.update(lo=None, hi=None, why=f"only {ok.size} of {n_boot} resamples gave a value")
        return out
    lo, hi = np.percentile(ok, [2.5, 97.5])
    out.update(lo=float(lo), hi=float(hi))
    return out


# =============================================================== models

def _design(t, t0, hinges, annual):
    t = np.asarray(t, float)
    cols = [np.ones_like(t), (t - t0) / 365.25] + [np.maximum(0.0, (t - h) / 365.25) for h in hinges]
    if annual:
        w = 2.0 * np.pi * t / 365.25
        cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols)


def fit_series(days, y, hinges=(), annual=True) -> dict | None:
    """Least squares of the protocol model: offset + rate (+ a slope change at
    each hinge: continuous piecewise linear) (+ annual sin, cos). A hinge
    with fewer than 2 epochs on either side is dropped (its segments merge).
    The covariance is inflated by the residuals' lag-1 autocorrelation r1
    (effective sample size n(1-r1)/(1+r1)), as the protocol fixes."""
    t, y = np.asarray(days, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size < 4:
        return None
    o = np.argsort(t, kind="stable")
    t, y = t[o], y[o]
    hs = sorted(float(h) for h in hinges if t[0] < h <= t[-1])
    while hs:
        edges = [-np.inf] + hs + [np.inf]
        cnt = [int(((t >= edges[i]) & (t < edges[i + 1])).sum()) for i in range(len(edges) - 1)]
        bad = next((k for k in range(len(hs)) if cnt[k] < 2 or cnt[k + 1] < 2), None)
        if bad is None:
            break
        hs.pop(bad)
    X = _design(t, t[0], hs, annual)
    p = X.shape[1]
    if t.size < p + 3:
        return None
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    s2 = float(res @ res) / (t.size - p)
    r1 = 0.0
    if t.size > 4 and np.std(res) > 0:
        r1 = float(np.corrcoef(res[:-1], res[1:])[0, 1])
        r1 = 0.0 if not math.isfinite(r1) else min(r1, 0.95)
    infl = (1 + r1) / (1 - r1) if r1 > 0 else 1.0
    cov = s2 * infl * np.linalg.pinv(X.T @ X)
    return {"t": t, "y": y, "t0": float(t[0]), "hinges": hs, "annual": annual, "coef": coef, "cov": cov,
            "res": res, "r1": r1, "n": int(t.size), "n_eff": float(t.size * (1 - max(r1, 0)) / (1 + max(r1, 0))),
            "rms": float(np.sqrt(np.mean(res ** 2)))}


def _slope_vec(fit, day) -> np.ndarray:
    c = np.zeros(len(fit["coef"]))
    c[1] = 1.0
    for k, h in enumerate(fit["hinges"]):
        if day >= h:
            c[2 + k] = 1.0
    return c


def _lin(fit, c):
    return float(c @ fit["coef"]), float(math.sqrt(max(float(c @ fit["cov"] @ c), 0.0)))


def model_rates(days, y, prm) -> dict:
    """{'rates': {label: {rate, sigma, n, span_yr}}, 'overlap_mean': {...}}
    in mm/yr, from the protocol model fitted to (days, y). With water years
    in the protocol (B-1): a rate for each water year in which the series has
    >= wy_min_epochs and >= wy_min_years, and none for the others. Without
    (B-2, stable ground): one linear rate over the matched record."""
    fit = fit_series(days, y, prm.get("hinge_days", []), prm.get("annual", True))
    if fit is None:
        return {"rates": {}, "overlap_mean": None, "fit": None}
    t = fit["t"]
    rates = {}
    if prm.get("water_years"):
        for label, a, b in prm["water_years"]:
            m = (t >= a) & (t < b)
            if m.sum() >= prm.get("wy_min_epochs", 10) and (t[m][-1] - t[m][0]) / 365.25 >= prm.get("wy_min_years", 0.75):
                r, s = _lin(fit, _slope_vec(fit, 0.5 * (t[m][0] + t[m][-1])))
                rates[label] = {"rate": r, "sigma": s, "n": int(m.sum()), "span_yr": float((t[m][-1] - t[m][0]) / 365.25)}
    else:
        span = (t[-1] - t[0]) / 365.25
        if span >= LINEAR_MIN_YEARS and t.size >= LINEAR_MIN_EPOCHS:
            r, s = _lin(fit, _slope_vec(fit, t[0]) if not fit["hinges"] else _slope_vec(fit, 0.5 * (t[0] + t[-1])))
            rates["linear"] = {"rate": r, "sigma": s, "n": int(t.size), "span_yr": float(span)}
    span = (t[-1] - t[0]) / 365.25
    om = None
    if span > 0:
        c = (_design(t[-1:], fit["t0"], fit["hinges"], fit["annual"])[0]
             - _design(t[:1], fit["t0"], fit["hinges"], fit["annual"])[0]) / span
        r, s = _lin(fit, c)
        om = {"rate": r, "sigma": s, "n": int(t.size), "span_yr": float(span)}
    return {"rates": rates, "overlap_mean": om,
            "fit": {"n": fit["n"], "r1": fit["r1"], "rms": fit["rms"], "hinges": [_iso(h) for h in fit["hinges"]]}}


# =============================================================== InSAR product

class InsarProduct:
    """The delivered MintPy products of one run, read cell by cell.

    Only the cells near the truth points are read from the time-series
    cube (deliverables._Cube streams them); the 2-D layers (temporal
    coherence, LOS geometry) are read whole once."""

    def __init__(self, cfg):
        import h5py
        from . import deliverables as dlv
        self.cfg, self._dlv = cfg, dlv
        self.ts_path = Path(dlv.timeseries_file(cfg))
        with h5py.File(self.ts_path, "r") as f:
            self.attrs = {k: _s(v) for k, v in f.attrs.items()}
            self.dates = [_s(d) for d in f["date"][:]]
        self.days = _days(self.dates)
        self.cube = dlv._Cube(self.ts_path, "timeseries")
        self.shape = tuple(self.cube.shape[1:])
        unit = str(self.attrs.get("UNIT", "m")).strip().lower()
        self.scale = {"m": 1000.0, "mm": 1.0, "cm": 10.0}.get(unit, 1000.0)
        self.crs = dlv._crs(self.attrs)
        tc_path = dlv._product(cfg, "temporalCoherence.h5")
        if not tc_path.exists():
            raise FileNotFoundError(f"No temporal coherence ({tc_path}): the protocol's reliability rule needs it")
        tc, _ = dlv._read(tc_path, "temporalCoherence")
        if tuple(tc.shape) != self.shape:
            raise ValueError(f"temporal coherence is {tc.shape}, the time series {self.shape}")
        self.tcoh = np.asarray(tc, float)
        self.tcoh_path = tc_path
        self.reliability = "temporal coherence (the product's own mask quantity)"
        if str(getattr(cfg, "backend", "")).lower() == "nisar":
            self._nisar_reliability(cfg)
        self._geom: dict = {}
        self.uncertainty = self._stated_uncertainty()

    def _nisar_reliability(self, cfg) -> None:
        """protocol_B1 reliability for NISAR GUNW: 'mean interferometric
        coherence over the pairs used >= 0.4 (temporal coherence is 1.0 by
        construction on a chain network)'. nisar_backend writes that mean as
        band 1 of nisar_gunw/coherence_gate.tif on its crop lattice (with a
        margin round the AOI); it is put on the product grid by its own
        transform and takes temporal coherence's place in every floor test
        (primary and sensitivity floors alike). Cells MintPy left empty stay
        empty. Without the file every imaged cell would count as reliable, so
        that is said in `reliability`."""
        gate = Path(cfg.project_dir) / "nisar_gunw" / "coherence_gate.tif"
        if not gate.exists():
            self.reliability = (f"temporal coherence, which is 1.0 by construction on a NISAR chain: {gate} is "
                                "missing, so every imaged cell counts as reliable (protocol_B1 wants the mean pair "
                                "coherence)")
            log.warning("NISAR run without %s: the reliability rule cannot be applied", gate)
            return
        import rasterio
        from . import lband
        with rasterio.open(gate) as src:
            mean = src.read(1).astype(float)
            t = src.transform
        src_grid = lband.Grid(float(t.c), float(t.f), float(t.a), float(t.e), mean.shape[1], mean.shape[0])
        try:
            dst_grid = lband.Grid.from_attrs(self.attrs, self.shape)
        except (KeyError, ValueError) as exc:
            self.reliability = (f"temporal coherence (1.0 on a NISAR chain): {gate} could not be put on the "
                                f"product grid ({exc}), so every imaged cell counts as reliable")
            log.warning("NISAR coherence gate not used: %s", exc)
            return
        on = lband.resample_to(mean, src_grid, dst_grid)
        on[~np.isfinite(self.tcoh)] = np.nan
        self.tcoh = on
        self.tcoh_path = gate
        self.reliability = ("mean interferometric coherence over the pairs used (NISAR chain; "
                            "nisar_gunw/coherence_gate.tif band 1, protocol_B1 reliability) in place of temporal "
                            "coherence")

    # ---- 2-D layers
    def geometry(self, name):
        if name not in self._geom:
            arr = self._dlv.geometry_field(self.cfg, name)
            if arr is not None and tuple(arr.shape) != self.shape:
                raise ValueError(f"geometry layer {name} is {arr.shape}, the time series {self.shape}")
            self._geom[name] = None if arr is None else np.asarray(arr, float)
        return self._geom[name]

    def _stated_uncertainty(self) -> dict:
        """The pipeline's own stated 1-sigma LOS rate uncertainty per cell:
        the empirical repeatability by temporal-coherence class it delivers
        (deliverables/uncertainty_model.json), else MintPy's formal
        velocityStd (labelled: a floor, not the uncertainty)."""
        p = Path(self.cfg.project_dir) / "deliverables" / "uncertainty_model.json"
        if p.exists():
            try:
                um = json.loads(p.read_text())
                classes = []
                for row in um.get("by_class", []):
                    lo, hi = (float(v) for v in str(row["tc"]).split("-"))
                    classes.append((lo, hi, float(row["empirical_mm_yr"])))
                if classes:
                    stale = str((um.get("key") or {}).get("file", "")) not in ("", str(self.ts_path))
                    return {"source": f"empirical rate repeatability by temporal-coherence class ({p})",
                            "kind": "empirical", "classes": sorted(classes), "stale": stale}
            except (ValueError, KeyError, OSError) as exc:
                log.warning("Could not read %s (%s)", p, exc)
        import h5py
        vp = self._dlv._product(self.cfg, "velocity.h5")
        if vp.exists():
            with h5py.File(vp, "r") as f:
                if "velocityStd" in f and tuple(f["velocityStd"].shape) == self.shape:
                    unit = _s(f.attrs.get("UNIT", "m/year")).lower()
                    k = 1000.0 if unit.startswith("m/") else 1.0
                    return {"source": f"MintPy formal velocityStd ({vp.name}): a floor, not the uncertainty",
                            "kind": "formal", "grid": f["velocityStd"][:].astype(float) * k}
        return {"source": "none: the product states no rate uncertainty", "kind": "none"}

    def stated_sigma(self, rows, cols) -> np.ndarray:
        u = self.uncertainty
        if u["kind"] == "empirical":
            tc = self.tcoh[rows, cols]
            out = np.full(tc.shape, np.nan)
            classes = u["classes"]
            for lo, hi, v in classes:
                out[(tc >= lo) & (tc < hi + (1e-9 if hi >= 1.0 else 0.0))] = v
            out[np.isnan(out) & np.isfinite(tc) & (tc < classes[0][0])] = classes[0][2]
            return out
        if u["kind"] == "formal":
            return u["grid"][rows, cols]
        return np.full(np.shape(rows), np.nan)

    # ---- cells near a point
    def cells_near(self, lat: float, lon: float, radius_m: float):
        """(rows, cols, distance m) of the cells whose centres lie within
        radius_m of (lat, lon), on a geographic or a projected grid."""
        return cells_near(self.attrs, self.shape, lat, lon, radius_m)

    def read_cells(self, rows, cols) -> np.ndarray:
        """(n_dates, k) displacement in mm."""
        if not np.size(rows):
            return np.empty((len(self.dates), 0))
        return self.cube[:, np.asarray(rows), np.asarray(cols)].astype(float) * self.scale

    def pixel_ids(self, rows, cols):
        return self._dlv.radar_pixel_ids(self.cfg, rows, cols, self.shape)

    def describe(self) -> dict:
        a = self.attrs
        return {"timeseries": str(self.ts_path), "temporal_coherence": str(self.tcoh_path),
                "reliability_quantity": self.reliability,
                "dates": len(self.dates), "first": self.dates[0], "last": self.dates[-1],
                "crs": self.crs, "grid": list(self.shape),
                "reference": {k: a.get(k) for k in ("REF_LAT", "REF_LON", "REF_Y", "REF_X", "REF_DATE")},
                "stated_rate_uncertainty": self.uncertainty["source"]
                + (" (WARNING: computed for another time-series file)" if self.uncertainty.get("stale") else "")}


def cells_near(attrs: dict, shape, lat: float, lon: float, radius_m: float):
    """(rows, cols, distance m) of the cells of a MintPy-style grid (X_FIRST,
    X_STEP, ... and EPSG / UTM_ZONE) whose centres lie within radius_m of
    (lat, lon): geodesic distance on a lat/lon grid, metres on a projected one."""
    from . import deliverables as dlv
    x0, dx, y0, dy = dlv._geotransform(attrs)
    crs = dlv._crs(attrs)
    ny, nx = shape
    if crs == "EPSG:4326":
        dlat = radius_m / 110_000.0 * 1.25
        dlon = dlat / max(math.cos(math.radians(lat)), 1e-6)
        px, py, padx, pady = lon, lat, dlon / abs(dx), dlat / abs(dy)
    else:
        from pyproj import Transformer
        px, py = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
        padx, pady = radius_m / abs(dx) * 1.25, radius_m / abs(dy) * 1.25
    cc, rc = (px - x0) / dx - 0.5, (py - y0) / dy - 0.5
    c0, c1 = max(0, int(math.floor(cc - padx)) - 1), min(nx - 1, int(math.ceil(cc + padx)) + 1)
    r0, r1 = max(0, int(math.floor(rc - pady)) - 1), min(ny - 1, int(math.ceil(rc + pady)) + 1)
    if c0 > c1 or r0 > r1:
        return np.array([], int), np.array([], int), np.array([], float)
    rr, cc_ = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    rr, cc_ = rr.ravel(), cc_.ravel()
    xc, yc = x0 + (cc_ + 0.5) * dx, y0 + (rr + 0.5) * dy
    if crs == "EPSG:4326":
        from pyproj import Geod
        _, _, d = Geod(ellps="WGS84").inv(np.full(xc.shape, lon), np.full(yc.shape, lat), xc, yc)
    else:
        d = np.hypot(xc - px, yc - py)
    m = np.asarray(d) <= radius_m
    return rr[m], cc_[m], np.asarray(d)[m]


def sample_up(path, points: dict, radius_m: float, min_cells: int) -> dict:
    """D1's decomposed Up (decompose.py: vertical_timeseries_decomposed.h5, a
    MintPy-style series in metres on its own common grid at its own epochs,
    NaN where not solved) at each point: the median of the solved cells
    within the radius at each epoch; fewer than min_cells = no value. The
    grid's cells are not traced to radar pixels, so only the cell count is
    applied. {'dates', 'days', 'path', 'stations': {id: {series, err, n_cells}}}"""
    import h5py
    path = Path(path)
    with h5py.File(path, "r") as f:
        key = "timeseries" if "timeseries" in f else "up"
        attrs = {k: _s(v) for k, v in f.attrs.items()}
        dates = [_s(d) for d in f["date"][:]]
        shape = tuple(f[key].shape[1:])
        scale = {"m": 1000.0, "mm": 1.0}.get(str(attrs.get("UNIT", "m")).lower(), 1000.0)
        out = {}
        for pid, pt in points.items():
            r, c, _ = cells_near(attrs, shape, pt["lat"], pt["lon"], radius_m)
            if not r.size:
                out[pid] = None
                continue
            r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
            blk = f[key][:, r0:r1, c0:c1].astype(float)[:, r - r0, c - c0] * scale
            n = np.isfinite(blk).sum(1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(blk, axis=1)
                spread = 1.4826 * np.nanmedian(np.abs(blk - med[:, None]), axis=1)
            ok = n >= min_cells
            out[pid] = {"series": np.where(ok, med, np.nan), "n_cells": n,
                        "err": np.where(ok, 1.2533 * spread / np.sqrt(np.maximum(n, 1)), np.nan)}
    return {"path": str(path), "dates": dates, "days": _days(dates), "stations": out,
            "quantity": attrs.get("DECOMPOSITION_QUANTITY", "vertical (Up)"),
            "north": attrs.get("DECOMPOSITION_NORTH", "north not estimated")}


def sample_points(prod, points: dict, radius_m: float, floors, min_cells: int, min_indep: int) -> dict:
    """{point id: {floor: sample}} - the protocol's sampling at every SAR date.

    sample = series (mm, NaN where the minimum is not met), n_cells,
    n_independent, spread (1.4826 x MAD, mm), iqr (mm), err (1.2533 x spread
    / sqrt(n_independent): the median's own sampling error, used only as a
    weight), los (unit vector, median over the reliable cells), incidence,
    stated_rate_sigma (median over the reliable cells), cells_within_radius,
    reliable_cells."""
    found = {}
    all_r, all_c = [], []
    for pid, pt in points.items():
        r, c, d = prod.cells_near(pt["lat"], pt["lon"], radius_m)
        found[pid] = (r, c, d)
        all_r.append(r)
        all_c.append(c)
    R = np.concatenate(all_r) if all_r else np.array([], int)
    C = np.concatenate(all_c) if all_c else np.array([], int)
    ids, how = prod.pixel_ids(R, C) if R.size else (np.array([], np.int64), "")
    le, ln, lu = (prod.geometry(k) for k in ("los_east", "los_north", "los_up"))
    if le is None or ln is None or lu is None:
        raise ValueError("The product has no per-pixel LOS geometry (deliverables.geometry_field): "
                         "the GNSS cannot be projected onto the line of sight")
    inc = prod.geometry("incidence")
    out, k0 = {}, 0
    for pid, (r, c, d) in found.items():
        pid_ids = ids[k0:k0 + r.size]
        k0 += r.size
        ts = prod.read_cells(r, c) if r.size else np.empty((len(prod.dates), 0))
        tc = prod.tcoh[r, c] if r.size else np.array([])
        stated = prod.stated_sigma(r, c) if r.size else np.array([])
        per = {}
        for fl in floors:
            rel = np.isfinite(tc) & (tc >= fl)
            v = ts[:, rel]
            fin = np.isfinite(v)
            n_cells = fin.sum(1)
            pids = pid_ids[rel]
            n_ind = np.array([np.unique(pids[fin[i]]).size for i in range(v.shape[0])]) if v.size else \
                np.zeros(len(prod.dates), int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(v, axis=1) if v.shape[1] else np.full(len(prod.dates), np.nan)
                spread = 1.4826 * np.nanmedian(np.abs(v - med[:, None]), axis=1) if v.shape[1] else med * np.nan
                q75, q25 = (np.nanpercentile(v, [75, 25], axis=1) if v.shape[1]
                            else (med * np.nan, med * np.nan))
            ok = (n_cells >= min_cells) & (n_ind >= min_indep)
            # the LOS vector is geometry, valid whether or not the cells are reliable: a masked
            # point still has a truth projected on its line of sight
            gsel = rel if rel.any() else np.ones(r.size, bool)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vec = np.array([np.nanmedian(g[r[gsel], c[gsel]]) if gsel.any() else np.nan for g in (le, ln, lu)])
            norm = float(np.linalg.norm(vec)) if np.all(np.isfinite(vec)) else float("nan")
            los = vec / norm if norm and math.isfinite(norm) else vec
            per[fl] = {
                "series": np.where(ok, med, np.nan), "n_cells": n_cells, "n_independent": n_ind,
                "spread": spread, "iqr": q75 - q25,
                "err": np.where(ok, 1.2533 * spread / np.sqrt(np.maximum(n_ind, 1)), np.nan),
                "los": los,
                "incidence": float(np.nanmedian(inc[r[gsel], c[gsel]])) if inc is not None and gsel.any() else None,
                "stated_rate_sigma": float(np.nanmedian(stated[rel])) if rel.any() and np.isfinite(stated[rel]).any()
                else None,
                "cells_within_radius": int(r.size), "reliable_cells": int(rel.sum()),
                "rows": r[rel], "cols": c[rel],
            }
        out[pid] = per
    out["_pixel_id_method"] = how
    return out


def coverage(prod, bbox, floors, per_epoch: bool = True) -> dict:
    """Share of the imaged cells inside the AOI box that are reliable, over
    the window (temporal coherence >= floor) and, optionally, per date
    (also finite at that date)."""
    dlv = prod._dlv
    box = dlv._box_mask(prod.attrs, bbox, prod.shape) if bbox else np.ones(prod.shape, bool)
    obs, how = dlv.observable_mask(prod.cfg, prod.shape)
    if obs is None:
        obs, how = np.isfinite(prod.tcoh) & (prod.tcoh > 0), "temporal coherence footprint"
    den = box & obs
    out = {"denominator": f"imaged cells inside the AOI box ({how})", "cells": int(den.sum()), "by_floor": {}}
    for fl in floors:
        rel = den & (prod.tcoh >= fl)
        out["by_floor"][str(fl)] = {"share_reliable": _f(rel.sum() / max(den.sum(), 1), 4)}
    if per_epoch and den.any():
        import h5py
        rows = np.flatnonzero(den.any(1))
        cols = np.flatnonzero(den.any(0))
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        sub = den[r0:r1, c0:c1]
        rel_sub = {fl: sub & (prod.tcoh[r0:r1, c0:c1] >= fl) for fl in floors}
        shares = {fl: [] for fl in floors}
        with h5py.File(prod.ts_path, "r") as f:
            d = f["timeseries"]
            step = int(d.chunks[0]) if d.chunks else 8
            for i in range(0, d.shape[0], step):
                blk = d[i:i + step, r0:r1, c0:c1]
                fin = np.isfinite(blk)
                for fl in floors:
                    shares[fl] += [float((fin[j] & rel_sub[fl]).sum() / max(sub.sum(), 1)) for j in range(blk.shape[0])]
        for fl in floors:
            s = np.array(shares[fl])
            out["by_floor"][str(fl)]["per_date"] = {"min": _f(s.min(), 4), "median": _f(np.median(s), 4),
                                                    "max": _f(s.max(), 4)}
            out["by_floor"][str(fl)]["_per_date_values"] = s
    return out


# =============================================================== ground truth

def _ground_truth():
    return importlib.import_module(f"{__package__}.ground_truth" if __package__ else "ground_truth")


# The digitised truth's sigma per component (protocol A amendment 2026-10-08/p23-1): the CSV column suffix of
# each ground_truth component, the terms the amended sigma adds in quadrature, and the sigma as first fixed
_COMP_SUFFIX = {"u": "", "e": "_east", "n": "_north"}
_SIGMA_TERMS = ("digitising_rms{s}_mm", "half_line_width{s}_mm", "calibration{s}_mm", "gnss_sigma{s}_mm")
_SIGMA_V1 = {"u": "sigma_up_as_first_fixed_mm", "e": "sigma_east_as_first_fixed_mm",
             "n": "sigma_north_as_first_fixed_mm"}


def _truth_columns(points) -> "object | None":
    """The CSV's own columns, indexed by (id, date), or None when the file
    cannot be read (a stand-in reader, a missing file)."""
    try:
        import pandas as pd
        raw = pd.read_csv(points, comment="#", dtype={"id": str}, skipinitialspace=True)
    except Exception:                                     # noqa: BLE001 - recorded by the caller
        return None
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "id" not in raw.columns or "date" not in raw.columns:
        return None
    raw["id"] = raw["id"].astype(str).str.strip()
    raw["_day"] = [_day(str(d).strip()[:10]) for d in raw["date"]]
    return raw.drop_duplicates(["id", "_day"]).set_index(["id", "_day"])


def digitised_truth_sigma(points, pts_truth, rule) -> tuple[list, dict]:
    """The truth sigma of each digitised point-epoch under the protocol's
    rule. rule 'hypot_repeatability_line_calibration' (amendment
    2026-10-08/p23-1): sqrt(digitising_rms^2 + half_line_width^2 +
    calibration^2 + gnss_sigma^2) per row and component, from the CSV's own
    columns; rule None (the protocol as first fixed): the CSV's
    sigma_*_as_first_fixed_mm (two-pass RMS at least the half-gridline).
    Where the columns a rule needs are missing, the file's sigma stays, and
    the record says so. Returns (points with the sigma replaced, record)."""
    rec = {"rule": rule or "as first fixed (the file's sigma_*_as_first_fixed_mm, else its sigma)",
           "replaced": 0, "kept_file_sigma": 0}
    if not pts_truth:
        return pts_truth, rec
    cols = _truth_columns(points) if points else None
    if cols is None:
        rec["why_kept"] = f"the truth file {points} could not be read for its sigma columns"
        rec["kept_file_sigma"] = sum(len(df) for _, df in pts_truth)
        return pts_truth, rec
    out, diffs = [], []
    for pt, df in pts_truth:
        df = df.copy()
        sig = df["sigma_mm"].to_numpy(float).copy()
        for i, (t, comp) in enumerate(zip(df["time"].values, df["component"])):
            key = (str(pt["id"]), _day(np.datetime64(t, "D")))
            s = _COMP_SUFFIX.get(str(comp))
            new = None
            if s is not None and key in cols.index:
                row = cols.loc[key]
                if rule == "hypot_repeatability_line_calibration":
                    terms = [row.get(n.format(s=s)) for n in _SIGMA_TERMS]
                    if all(v is not None and np.isfinite(float(v)) for v in terms):
                        new = float(np.sqrt(np.sum(np.square(np.asarray(terms, float)))))
                elif rule is None:
                    v = row.get(_SIGMA_V1[str(comp)])
                    if v is not None and np.isfinite(float(v)):
                        new = float(v)
            if new is not None and new > 0:
                diffs.append(abs(new - sig[i]))
                sig[i] = new
                rec["replaced"] += 1
            else:
                rec["kept_file_sigma"] += 1
        df["sigma_mm"] = sig
        out.append((pt, df))
    if rule is not None and rec["kept_file_sigma"]:
        rec["why_kept"] = ("the rows without the columns the rule needs ("
                           + ", ".join(n.format(s="[_east|_north]") for n in _SIGMA_TERMS) + ") keep the file's sigma")
    rec["max_change_from_file_sigma_mm"] = _f(max(diffs), 2) if diffs else None
    up = [float(v) for _, df in out for v in df.loc[df["component"] == "u", "sigma_mm"]]
    if up:
        rec["up_sigma_mm"] = {"min": _f(min(up), 2), "median": _f(float(np.median(up)), 2), "max": _f(max(up), 2)}
    return out, rec


def split_truth_units(pts_truth, splits) -> tuple[list, list]:
    """Amendment 2026-10-08/p23-4: a unit relocated part-way (Tahmoor West
    Site 13, early 2022) is scored as two units. Each split {id, from, as}
    moves the rows of `id` dated on or after `from` to a new point `as`,
    which takes its own first epoch as its datum; no change across the
    relocation is ever formed. Returns (points, what was split)."""
    done = []
    if not pts_truth or not splits:
        return pts_truth, done
    by = {str(s.get("id")): s for s in splits if isinstance(s, dict) and s.get("id") and s.get("from") and s.get("as")}
    out = []
    for pt, df in pts_truth:
        s = by.get(str(pt["id"]))
        if s is None:
            out.append((pt, df))
            continue
        cut = np.datetime64(str(s["from"])[:10])
        t = df["time"].values.astype("datetime64[D]")
        before, after = df[t < cut].reset_index(drop=True), df[t >= cut].reset_index(drop=True)
        for d in (before, after):
            d.attrs.update(df.attrs)
        rec = {"id": str(pt["id"]), "from": str(s["from"])[:10], "as": str(s["as"]),
               "rows_before": int(len(before)), "rows_after": int(len(after))}
        if len(before):
            out.append((pt, before))
        if len(after):
            pt2 = dict(pt)
            pt2["id"] = str(s["as"])
            after.attrs["id"] = str(s["as"])
            out.append((pt2, after))
        done.append(rec)
    return out, done


_PROV_KEYS = ("source_url", "file", "sha256", "fetched", "fetch_note", "first_epoch_in_file", "last_epoch_in_file",
              "ends_before_window_end", "first_day", "last_day", "n_days", "span_yr", "frame", "steps_removed",
              "steps_not_removed", "earthquake_steps_removed", "outliers_removed", "outlier_sigma",
              "robust_sigma_mm", "model", "sigma_rule", "datum", "steps_note", "stated_sigma_median_mm")


def gnss_from_frame(sid: str, df, source: str) -> dict:
    """One station's daily E/N/U (mm) from a ground_truth DataFrame, with the
    provenance ground_truth recorded (source URL, sha256 and fetch date of
    the file, its last epoch, the steps and outliers removed, the screening
    model). The harness does no screening of its own: ground_truth already
    removed the steps and the daily outliers (protocol gnss_screening)."""
    import pandas as pd
    idx = pd.DatetimeIndex(df.index)
    days = idx.values.astype("datetime64[D]").astype(np.int64)
    enu = df[["e_mm", "n_mm", "u_mm"]].to_numpy(float)
    have = all(c in df.columns for c in ("se_mm", "sn_mm", "su_mm"))
    sig = df[["se_mm", "sn_mm", "su_mm"]].to_numpy(float) if have else np.full_like(enu, np.nan)
    o = np.argsort(days, kind="stable")
    days, enu, sig = days[o], enu[o], sig[o]
    keep = np.r_[True, np.diff(days) > 0]
    days, enu, sig = days[keep], enu[keep], sig[keep]
    attrs = dict(getattr(df, "attrs", {}) or {})
    prov = {"source": source, **{k: attrs[k] for k in _PROV_KEYS if k in attrs}}
    model = attrs.get("model")
    name = model.get("name") if isinstance(model, dict) else model
    hinges = (model.get("hinges") if isinstance(model, dict) else None) or []
    rs = attrs.get("robust_sigma_mm") or {}
    return {"id": sid, "source": source, "days": days, "enu": enu, "formal_sigma": sig, "provenance": prov,
            "lat": attrs.get("lat"), "lon": attrs.get("lon"), "model": name or "linear",
            "hinge_days": [_day(h) for h in hinges],
            "sigma": np.array([float(rs.get(c, np.nan)) for c in "enu"])}


def _screen_models(prm) -> tuple[dict, list]:
    """The screening model ground_truth applies to each station, from the
    protocol: B1 piecewise by water year, B2 and stable stations linear
    (+ annual), an event station (protocol A's validation stations) a local
    line - a rate model would call a panel event an outlier."""
    if prm["design"] == "network":
        m = "piecewise" if prm.get("hinges") else "linear"
        return {s: m for s in prm["stations"]}, list(prm.get("hinges") or [])
    return {s: ("local" if s in prm.get("validation_stations", []) else "linear") for s in prm["stations"]}, []


def load_ngl(ids, window, cache_dir, prm, gt=None) -> tuple[dict, dict]:
    """NGL IGS20 series of the stations ({id: station}, {id: why not loaded})."""
    gt = gt or _ground_truth()
    models, hinges = _screen_models(prm)
    try:
        res = gt.load_gnss_series(list(ids), window=window, cache_dir=cache_dir, frame="IGS20",
                                  model={s: models.get(s, "linear") for s in ids}, hinges=hinges,
                                  earthquake_steps=bool(prm.get("earthquake_steps", False)))
    except Exception as exc:                              # noqa: BLE001 - recorded, the run says so
        return {}, {s: f"NGL series not loaded: {exc}" for s in ids}
    bad = dict(getattr(res, "missing", {}) or {})
    out = {}
    for sid in ids:
        df = res.get(sid)
        if df is None or len(df) == 0:
            bad.setdefault(sid, "NGL series has no days inside the window")
            continue
        out[sid] = gnss_from_frame(sid, df, "ngl")
    return out, bad


def load_ga(path, ids, window, prm, gt=None) -> tuple[dict, dict]:
    """Geoscience Australia CEMP daily E/N/U from the xlsx, for the stations
    the protocol names (coordinates from the protocol: the workbook has none)."""
    gt = gt or _ground_truth()
    models, _ = _screen_models(prm)
    coords = {s: (prm["stations"][s]["lat"], prm["stations"][s]["lon"]) for s in ids}
    res = gt.load_ga_xlsx(path, sites=list(ids), window=window, coords=coords,
                          model={s: models.get(s, "linear") for s in ids})
    bad = {s: f"GA: {w}" for s, w in (getattr(res, "missing", {}) or {}).items()}
    out = {s: gnss_from_frame(s, df, "ga") for s, df in res.items() if df is not None and len(df)}
    return out, bad


def _model_curve(days, y, model="linear", hinge_days=(), annual=True):
    """The station's screening model fitted to a projected daily series
    (for its robust sigma): 'local' a running median over 31 days, else the
    protocol model."""
    if model == "local":
        return _running_median(days, y, SCREEN_RUNNING_MEDIAN_DAYS)
    fit = fit_series(days, y, hinge_days if model == "piecewise" else (), annual)
    if fit is None:
        return _running_median(days, y, SCREEN_RUNNING_MEDIAN_DAYS)
    return _design(np.asarray(days, float), fit["t0"], fit["hinges"], fit["annual"]) @ fit["coef"]


def _running_median(days, y, width):
    import pandas as pd
    s = pd.Series(y, index=pd.to_datetime(np.asarray(days, "datetime64[D]")))
    return s.rolling(f"{int(width)}D", center=True, min_periods=3).median().to_numpy()


def _robust_sigma(res) -> float:
    r = np.asarray(res, float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def gnss_daily_component(st, vec) -> np.ndarray:
    """Daily series of the station projected on vec (3,) or one component."""
    return st["enu"] @ np.asarray(vec, float)


def formal_sigma_days(st) -> tuple[np.ndarray | None, str]:
    """The source's own formal 1-sigma per day (n_days, 3: E, N, U in mm) for
    the days the station kept, and where it came from. ground_truth hands the
    harness each day's sigma already as the LARGER of the formal sigma and
    the robust sigma of that component, so the formal part is read again
    from the source: the NGL tenv3 file named in the provenance (its se, sn,
    su columns), or GA's median stated sigma. When neither can be had, the
    per-day sigma ground_truth gave is used and labelled an upper bound
    (amendment 2026-10-08/4 then takes a value that is at least as large as
    the rule's)."""
    prov = st.get("provenance") or {}
    days = np.asarray(st.get("days", []), np.int64)
    if st.get("source") == "ngl" and prov.get("file") and Path(str(prov["file"])).exists():
        try:
            from .gnss_tie import read_tenv3
            g = read_tenv3(Path(str(prov["file"])), full=True)
            gd = np.array([np.datetime64(d, "D").astype(np.int64) for d in g["date"]], np.int64)
            sig = np.column_stack([g["se"], g["sn"], g["su"]]).astype(float)
            o = np.argsort(gd, kind="stable")
            gd, sig = gd[o], sig[o]
            j = np.clip(np.searchsorted(gd, days), 0, max(gd.size - 1, 0))
            hit = gd.size > 0
            out = np.full((days.size, 3), np.nan)
            if hit:
                m = gd[j] == days
                out[m] = sig[j[m]]
            if np.isfinite(out).any():
                return out, f"NGL formal sigma (tenv3 se/sn/su columns of {Path(str(prov['file'])).name})"
        except Exception as exc:                          # noqa: BLE001 - falls back, and says so
            log.warning("%s: the tenv3 formal sigmas were not read (%s)", st.get("id"), exc)
    med = prov.get("stated_sigma_median_mm")
    if st.get("source") == "ga" and isinstance(med, dict) and all(c in med for c in "enu"):
        return (np.tile(np.array([float(med[c]) for c in "enu"]), (days.size, 1)),
                "GA's stated sigma, median over the window (the workbook's sigma columns)")
    fs = st.get("formal_sigma")
    if fs is not None and np.size(fs):
        return (np.asarray(fs, float), "upper bound: the per-day sigma ground_truth gave (already the larger of "
                                       "the formal and the robust sigma of each component); the formal sigma "
                                       "alone could not be read")
    return None, "no formal sigma"


def daily_sigma_record(st, vec, prm) -> dict:
    """The daily sigma of a station in one component (a unit vector, or a
    LOS vector), and how it was had: the robust sigma of the projected daily
    series about the station's screening model (the protocols as first
    fixed), and the source's formal sigma (the per-day sigmas projected onto
    the component, median over the days; st['formal_raw'] when the harness
    read it from the source, else formal_sigma_days). Amendment 2026-10-08/4
    takes the larger of the two; without it the robust sigma is used. Also
    the robust range (p95 - p5): the signal for the series-correlation rule."""
    nan = float("nan")
    rec = {"robust_mm": nan, "formal_mm": nan, "used_mm": nan, "range_mm": nan, "rule": "robust"}
    if not np.all(np.isfinite(vec)):
        return rec
    y = gnss_daily_component(st, vec)
    if y.size < 5:
        return rec
    s = _robust_sigma(y - _model_curve(st["days"], y, st.get("model", "linear"), st.get("hinge_days", ()),
                                       prm.get("annual", True)))
    p5, p95 = np.nanpercentile(y, [5, 95])
    if "formal_raw" not in st:                            # read once per station, then kept with it
        st["formal_raw"], st["formal_source"] = formal_sigma_days(st)
    fs = st.get("formal_raw")
    fs = np.asarray(fs, float) if fs is not None else np.empty((0, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_day = np.sqrt(((fs * np.asarray(vec, float)[None, :]) ** 2).sum(1)) if fs.ndim == 2 else np.array([])
        formal = float(np.nanmedian(per_day)) if per_day.size and np.isfinite(per_day).any() else nan
    rec.update(robust_mm=s, formal_mm=formal, range_mm=float(p95 - p5), used_mm=s,
               formal_source=st.get("formal_source"))
    if (prm.get("amend") or {}).get("gnss_daily_sigma") == "larger_of_robust_and_formal":
        rec["rule"] = "larger of robust and formal (amendment 2026-10-08/4)"
        if math.isfinite(formal) and (not math.isfinite(s) or formal > s):
            rec["used_mm"], rec["used"] = formal, "formal"
        else:
            rec["used"] = "robust"
    else:
        rec["used"] = "robust"
    return rec


def daily_sigma_and_range(st, vec, prm) -> tuple[float, float]:
    """The station's daily sigma in the component (daily_sigma_record: robust,
    or under amendment 2026-10-08/4 the larger of robust and formal), and its
    robust range (p95 - p5): the signal for the series-correlation rule."""
    r = daily_sigma_record(st, vec, prm)
    return r["used_mm"], r["range_mm"]


def match_epochs(st, sar_days, half_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean daily E/N/U within +-half_days of each SAR date (NaN where none),
    and the number of days averaged."""
    d = st["days"]
    sar = np.asarray(sar_days, np.int64)
    lo = np.searchsorted(d, sar - half_days, side="left")
    hi = np.searchsorted(d, sar + half_days, side="right")
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(st["enu"], axis=0)])
    n = hi - lo
    mean = (cs[hi] - cs[lo]) / np.maximum(n, 1)[:, None]
    mean[n == 0] = np.nan
    return mean, n


# =============================================================== network design (protocols B1, B2)

def _rate_key_lists(prm, P):
    """Which rates the rows compare: water-year rates (B1) or one linear rate
    (B2); and which feeds the uncertainty-calibration row (B1 overlap mean,
    B2 the linear rate)."""
    uc = str(((P.get("comparisons") or {}).get("uncertainty_calibration") or {}).get("what", ""))
    return ("overlap_mean" if "overlap-mean" in uc or "overlap mean" in uc else "primary")


def _loo_datum(cal, Y, Wt, n_t, min_stations=2):
    """Per-epoch offset o(t) and one constant b_c per calibration station
    from y_c(t) = o(t) + b_c (weights Wt), b of the station with the longest
    record fixed to 0. Only epochs with >= min_stations calibration values,
    and only the stations tied to that station through shared epochs."""
    cal = [c for c in cal if np.isfinite(Y[c]).any()]
    info = {"stations": [], "epochs": 0, "constant_fixed_at": None, "untied": []}
    o = np.full(n_t, np.nan)
    if len(cal) < min_stations:
        info["why"] = f"{len(cal)} calibration station(s) with values; at least {min_stations} are needed"
        return o, {}, info
    F = np.array([np.isfinite(Y[c]) & np.isfinite(Wt[c]) for c in cal])
    while True:
        ep = F.sum(0) >= min_stations
        Fe = F & ep[None, :]
        has = Fe.any(1)
        if not has.any():
            info["why"] = f"no epoch has {min_stations} calibration stations with values"
            return o, {}, info
        counts = Fe.sum(1)
        c0 = int(np.argmax(np.where(has, counts, -1)))
        # stations connected to c0 through shared epochs
        comp, frontier = {c0}, {c0}
        while frontier:
            eps = Fe[list(frontier)].any(0)
            new = {i for i in np.flatnonzero((Fe[:, eps]).any(1)) if i not in comp}
            comp |= new
            frontier = new
        drop = [i for i in range(len(cal)) if has[i] and i not in comp]
        if not drop:
            break
        info["untied"] += [cal[i] for i in drop]
        F[drop] = False
    sts = sorted(comp)
    eidx = np.flatnonzero(ep)
    col_e = {e: k for k, e in enumerate(eidx)}
    bst = [i for i in sts if i != c0]
    col_b = {i: len(eidx) + k for k, i in enumerate(bst)}
    rows, rhs, w = [], [], []
    for i in sts:
        for e in np.flatnonzero(Fe[i]):
            r = np.zeros(len(eidx) + len(bst))
            r[col_e[e]] = 1.0
            if i in col_b:
                r[col_b[i]] = 1.0
            rows.append(r)
            rhs.append(Y[cal[i]][e])
            w.append(math.sqrt(Wt[cal[i]][e]))
    A, b, w = np.array(rows), np.array(rhs), np.array(w)
    sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    o[eidx] = sol[:len(eidx)]
    consts = {cal[c0]: 0.0}
    consts.update({cal[i]: float(sol[col_b[i]]) for i in bst})
    info.update(stations=[cal[i] for i in sts], epochs=int(len(eidx)), constant_fixed_at=cal[c0])
    return o, consts, info


def _plane_datum(cal, Y, consts, xy, k_xy, n_t, min_stations=4):
    """Sensitivity: per-epoch plane o + a x + b y through y_c(t) - b_c of the
    calibration stations where >= min_stations have values, evaluated at the
    scored station; NaN elsewhere."""
    o = np.full(n_t, np.nan)
    cal = [c for c in cal if c in consts]
    if len(cal) < min_stations:
        return o
    Yc = np.array([Y[c] - consts[c] for c in cal])
    X = np.array([[1.0, xy[c][0], xy[c][1]] for c in cal])
    for t in range(n_t):
        m = np.isfinite(Yc[:, t])
        if m.sum() >= min_stations:
            coef, *_ = np.linalg.lstsq(X[m], Yc[m, t], rcond=None)
            o[t] = coef @ np.array([1.0, k_xy[0], k_xy[1]])
    return o


def _significance():
    return importlib.import_module(f"{__package__}.significance" if __package__ else "significance")


def _joint_plane_datum(cal, Y, Wt, xy, k_xy, sar_days, min_stations=4):
    """Sensitivity beside the protocol's plane (_plane_datum): the same
    offset + plane per epoch, but fitted jointly with the station constants
    by significance.gnss_datum_fit (plane_method 'joint', the B1 code's
    gnss_absolute fit), evaluated at the scored station. The protocol's
    two-stage plane takes the constants from the offset-only fit, which
    biases the plane when stations join or leave (a 0.5 mm/yr/km tilt read
    0.28 in significance's test); this one does not. As first fixed it is
    reported, never graded; under amendment 2026-10-08/3 (B1, B2) it is the
    graded plane_check. Returns (per-epoch correction, why-not or None)."""
    cal = [c for c in cal if np.isfinite(Y[c]).any()]
    n_t = len(sar_days)
    if len(cal) < min_stations:
        return np.full(n_t, np.nan), f"{len(cal)} calibration station(s) with values; the plane needs {min_stations}"
    sg = _significance()
    d = np.array([Y[c] for c in cal], float)
    sig = np.array([1.0 / np.sqrt(np.asarray(Wt[c], float)) for c in cal], float)
    xyc = np.array([xy[c] for c in cal], float)
    try:
        fit = sg.gnss_datum_fit(np.asarray(sar_days, float) / 365.25, d, sigma=sig, xy=xyc, plane=True,
                                min_per_epoch_plane=min_stations, plane_method="joint")
    except ValueError as exc:
        return np.full(n_t, np.nan), str(exc)
    return np.asarray(sg.datum_correction(fit, k_xy), float), None


def _insar_epoch_sigma(days, y, prm) -> float:
    """Robust sigma of a station's InSAR series about the protocol rate model:
    the InSAR error of one epoch in the amended York fit (2026-10-08/2)."""
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    d = np.asarray(days)[m]
    model = "piecewise" if prm.get("hinge_days") else "linear"
    return _robust_sigma(y[m] - _model_curve(d, y[m], model, prm.get("hinge_days", ()), prm.get("annual", True)))


def interval_changes(sar_days, I, G, sgG, prm, stations) -> list[dict]:
    """Amendment 2026-10-08/2: for each interval of prm['change_intervals']
    (B1's water years, B2's years from the window start), the two SAR dates
    t0 < t1 inside it, at least prm['change_min_years'] apart, at which the
    most stations have both an InSAR and a GNSS value (ties: the longer span,
    then the earlier t0), and every such station's InSAR and GNSS change
    between those same two dates. One pair of dates for all stations, so the
    run's reference pixel moves by the same amount in every station's change."""
    sar = np.asarray(sar_days, np.int64)
    min_d = float(prm.get("change_min_years", 0.75)) * 365.25
    out = []
    for label, a, b in prm.get("change_intervals") or []:
        rec = {"interval": label, "from": _iso(a), "to": _iso(b - 1)}
        idx = np.flatnonzero((sar >= a) & (sar < b))
        if idx.size < 2 or not stations:
            out.append({**rec, "why": f"{idx.size} SAR date(s) in the interval"})
            continue
        ok = np.array([np.isfinite(I[s][idx]) & np.isfinite(G[s][idx]) for s in stations])
        cnt = ok.T.astype(np.int64) @ ok.astype(np.int64)                # [i, j]: stations with both dates
        span = (sar[idx][None, :] - sar[idx][:, None]).astype(float)       # [i, j]: day_j - day_i
        cnt = np.where(span >= min_d, cnt, -1)
        best = int(cnt.max())
        if best < 2:
            out.append({**rec, "why": (f"no two SAR dates {prm.get('change_min_years', 0.75)} yr apart with at "
                                       f"least 2 stations having values at both" if best >= 0 else
                                       f"the SAR dates span less than {prm.get('change_min_years', 0.75)} yr")})
            continue
        cand = np.argwhere(cnt == best)
        spans = span[cand[:, 0], cand[:, 1]]
        i, j = cand[np.lexsort((cand[:, 0], -spans))[0]]
        t0, t1 = int(idx[i]), int(idx[j])
        rows = []
        for k, s in enumerate(stations):
            if ok[k, i] and ok[k, j]:
                rows.append({"station": s, "insar": float(I[s][t1] - I[s][t0]), "gnss": float(G[s][t1] - G[s][t0]),
                             "s_gnss": float(math.hypot(np.asarray(sgG[s], float)[t0], np.asarray(sgG[s], float)[t1]))})
        out.append({**rec, "t0": _iso(sar[t0]), "t1": _iso(sar[t1]), "span_yr": _f((sar[t1] - sar[t0]) / 365.25, 3),
                    "stations": rows})
    return out


def change_slope(intervals, s_insar_epoch: dict, n_boot: int, seed: int) -> dict:
    """York slope of the per-station interval changes (InSAR on GNSS), each
    interval's mean over its stations removed from both sides (amendment
    2026-10-08/2): the reference pixel's change, the same for every station
    in an interval, cancels, and no station calibrates another. Errors:
    the GNSS change's (sqrt of the two dates' daily sigmas squared) and
    sqrt(2) x the station's InSAR epoch sigma. 95% CI: bootstrap over
    stations, each resample removing the interval means again (a station
    drawn twice counts twice in the mean)."""
    ivs = [iv for iv in intervals if len(iv.get("stations") or []) >= 2]
    st_list = sorted({r["station"] for iv in ivs for r in iv["stations"]})

    def build(counts):
        X, Y, SX, SY = [], [], [], []
        for iv in ivs:
            rows = [r for r in iv["stations"] if counts.get(r["station"], 0) > 0]
            if len(rows) < 2:
                continue
            w = np.array([counts[r["station"]] for r in rows], float)
            dI = np.array([r["insar"] for r in rows])
            dG = np.array([r["gnss"] for r in rows])
            mI, mG = float((w * dI).sum() / w.sum()), float((w * dG).sum() / w.sum())
            for r, n in zip(rows, w.astype(int)):
                sy = math.sqrt(2.0) * float(s_insar_epoch.get(r["station"], float("nan")))
                for _ in range(n):
                    X.append(r["gnss"] - mG)
                    Y.append(r["insar"] - mI)
                    SX.append(r["s_gnss"])
                    SY.append(sy)
        return np.array(X), np.array(Y), np.array(SX), np.array(SY)

    x, y, sx, sy = build({s: 1 for s in st_list})
    fit = york_fit(x, y, sx, sy) if x.size >= 3 else None

    def boot_slope(idx, _r):
        c = np.bincount(idx, minlength=len(st_list))
        f = york_fit(*build({s: int(c[k]) for k, s in enumerate(st_list)}))
        return f["slope"] if f else np.nan

    ci = bootstrap_ci(len(st_list), boot_slope, n_boot, seed) if fit else {"lo": None, "hi": None}
    inc1 = (ci["lo"] <= 1.0 <= ci["hi"]) if ci.get("lo") is not None else None
    why = None
    if fit is None:
        why = (f"{x.size} station-interval change(s) with at least 2 stations in their interval: the York fit "
               f"needs 3" if x.size < 3 else "the changes do not vary (no fit)")
    return {"what": "per-station interval changes, InSAR against GNSS (LOS), each interval's mean over its stations "
                    "removed from both (amendment 2026-10-08/2)",
            "n": int(x.size), "stations": st_list, "fit": fit, "ci_slope": ci, "ci_includes_1": inc1,
            "r": pearson(x, y), "why": why,
            "intervals": [{k: v for k, v in iv.items()} for iv in intervals],
            "insar_epoch_sigma_mm": {s: _f(v, 2) for s, v in s_insar_epoch.items()}}


def _datum_scores(series: dict, G: dict, sar_days, rate_ids, series_ids, stated, prm, P, sig_range,
                  n_boot: int, seed: int) -> dict:
    """The leave-one-out rows (bias, series RMSE and correlation, uncertainty
    calibration) in another datum: the per-station InSAR series already
    carrying that datum ({station: series}). Used for the graded plane check
    of amendment 2026-10-08/3 (the joint offset + plane fit)."""
    uc_key = _rate_key_lists(prm, P)
    per, rate_rows, unc_rows = {}, [], []
    for k in series_ids:
        A = series.get(k)
        if A is None:
            continue
        m = np.isfinite(A) & np.isfinite(G[k])
        row = {"epochs": int(m.sum())}
        if m.sum() >= 3:
            diff = (A - G[k])[m]
            res = diff - diff.mean()
            row.update(rmse=float(np.sqrt(np.mean(res ** 2))), r=pearson(A[m], G[k][m]))
            if sig_range and k in sig_range:
                s, rg = sig_range[k]
                row["qualifies_for_correlation"] = bool(math.isfinite(s) and math.isfinite(rg) and rg > 10 * s)
        if k in rate_ids and m.sum() >= 4:
            t = np.asarray(sar_days)[m]
            rA, rG = model_rates(t, A[m], prm), model_rates(t, G[k][m], prm)
            for lab in rA["rates"]:
                if lab in rG["rates"]:
                    rate_rows.append({"station": k, "rate": lab, "error": rA["rates"][lab]["rate"] - rG["rates"][lab]["rate"]})
            src_A = rA["overlap_mean"] if uc_key == "overlap_mean" else next(iter(rA["rates"].values()), None)
            src_G = rG["overlap_mean"] if uc_key == "overlap_mean" else next(iter(rG["rates"].values()), None)
            if src_A and src_G and stated.get(k) is not None:
                e = src_A["rate"] - src_G["rate"]
                lim = 1.96 * math.hypot(stated[k], src_G["sigma"])
                unc_rows.append({"station": k, "error": e, "limit": lim, "inside": abs(e) <= lim})
        per[k] = row
    by_st = {}
    for r_ in rate_rows:
        by_st.setdefault(r_["station"], []).append(r_["error"])
    st_list = list(by_st)
    errs = np.array([r_["error"] for r_ in rate_rows], float)
    es = error_stats(errs)
    ci_b = (bootstrap_ci(len(st_list), lambda idx, _r: float(np.mean(np.concatenate([by_st[st_list[i]] for i in idx]))),
                         n_boot, seed) if es.get("n") else {"lo": None, "hi": None})
    rm = {k: v["rmse"] for k, v in per.items() if "rmse" in v}
    rv = np.array(list(rm.values()))
    qual = {k: v["r"] for k, v in per.items() if v.get("qualifies_for_correlation") and math.isfinite(v.get("r", np.nan))}
    qv = np.array(list(qual.values()))
    n_in = sum(1 for u in unc_rows if u["inside"])
    return {
        "per_station": {k: {kk: (_f(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in per.items()},
        "rate_bias_loo": {"abs_bias": abs(es["bias"]) if es.get("n") else None, "bias": es.get("bias"),
                          "n": es.get("n", 0), "ci_bias": ci_b},
        "series_rmse": {"median_rmse": float(np.median(rv)) if rv.size else None,
                        "per_station": {k: _f(v, 2) for k, v in rm.items()},
                        "ci_median": bootstrap_ci(len(rv), lambda idx, _r: float(np.median(rv[idx])), n_boot, seed)
                        if rv.size else {"lo": None, "hi": None}},
        "series_correlation": {"r": float(np.median(qv)) if qv.size else None, "n_qualifying": len(qv),
                               "per_station": {k: _f(v, 3) for k, v in qual.items()}},
        "uncertainty_calibration": {"share_inside": n_in / len(unc_rows) if unc_rows else None,
                                    "n_outside": len(unc_rows) - n_in if unc_rows else None, "n": len(unc_rows),
                                    "rate_used": uc_key},
    }


def _local_xy_km(points: dict) -> dict:
    lat0 = float(np.mean([p["lat"] for p in points.values()]))
    lon0 = float(np.mean([p["lon"] for p in points.values()]))
    k = math.cos(math.radians(lat0))
    return {s: ((p["lon"] - lon0) * 111.32 * k, (p["lat"] - lat0) * 110.57) for s, p in points.items()}


def network_component(sar_days, I, G, sgI, sgG, stated, prm, P, xy, sig_range=None, n_boot=None, seed=None) -> dict:
    """Every network-design statistic for one component (LOS, or a vertical).

    I, G: {station: series at the SAR dates (mm)} - InSAR and truth in the
    same component; sgI: {station: per-date sigma of the InSAR value};
    sgG: {station: per-date truth sigma (daily robust sigma)}; stated:
    {station: the product's stated 1-sigma rate uncertainty}; sig_range:
    {station: (daily robust sigma, robust range)} for the correlation rule."""
    n_boot = n_boot or prm["n_boot"]
    seed = seed if seed is not None else prm["seed"]
    n_t = len(sar_days)
    rate_ids = [s for s in prm["rate_stations"] if s in I]
    series_ids = [s for s in prm["series_stations"] if s in I] or rate_ids
    pool = [s for s in prm["calibration_pool"] if s in I] or list(I)
    wy = bool(prm.get("water_years"))
    uc_key = _rate_key_lists(prm, P)
    out = {"stations_rate": rate_ids, "stations_series": series_ids, "calibration_pool": pool}

    # ---------------- station-pair double differences
    pairs = []
    for i, j in combinations(rate_ids, 2):
        m = np.isfinite(I[i]) & np.isfinite(I[j]) & np.isfinite(G[i]) & np.isfinite(G[j])
        t = sar_days[m]
        rI = model_rates(t, (I[i] - I[j])[m], prm)
        rG = model_rates(t, (G[i] - G[j])[m], prm)
        errs = {lab: rI["rates"][lab]["rate"] - rG["rates"][lab]["rate"]
                for lab in rI["rates"] if lab in rG["rates"]}
        pairs.append({"pair": f"{i}-{j}", "i": i, "j": j, "epochs": int(m.sum()),
                      "insar": {k: _f(v["rate"], 2) for k, v in rI["rates"].items()},
                      "gnss": {k: _f(v["rate"], 2) for k, v in rG["rates"].items()},
                      "error": {k: _f(v, 2) for k, v in errs.items()}, "_err": list(errs.values())})
    all_dd = np.array([e for p in pairs for e in p["_err"]], float)
    dd = error_stats(all_dd)
    sidx = {s: k for k, s in enumerate(rate_ids)}
    S = len(rate_ids)
    SS, NN = np.zeros((S, S)), np.zeros((S, S))
    for p in pairs:
        a, b = sidx[p["i"]], sidx[p["j"]]
        e = np.asarray(p["_err"], float)
        SS[a, b] = SS[b, a] = float(np.sum(e ** 2))
        NN[a, b] = NN[b, a] = float(e.size)

    def dd_rmse(idx, _rng):
        c = np.bincount(idx, minlength=S).astype(float)
        w = np.triu(np.outer(c, c), 1)
        den = float(np.sum(w * NN))
        return math.sqrt(float(np.sum(w * SS)) / den) if den > 0 else np.nan

    dd_ci = bootstrap_ci(S, dd_rmse, n_boot, seed) if dd.get("n") else {"lo": None, "hi": None}
    why = {}
    if len(rate_ids) < 2:
        why["dd"] = f"{len(rate_ids)} rate station(s) with an InSAR value: a double difference needs two"
    elif not dd.get("n"):
        best = max(pairs, key=lambda p_: p_["epochs"])
        why["dd"] = (f"no station pair has a rate over a common record: the longest common record is "
                     f"{best['epochs']} matched epochs ({best['pair']})")
    out["dd"] = {"stats": dd, "ci_rmse": dd_ci, "pairs": [{k: v for k, v in p.items() if k != "_err"} for p in pairs]}

    # ---------------- GNSS-absolute leave-one-out
    Y = {c: G[c] - I[c] for c in pool}
    # weights 1/sigma^2; a sigma floor of 0.1 mm keeps a noise-free value from taking infinite weight
    Wt = {c: 1.0 / np.maximum(np.asarray(sgG[c], float) ** 2 + np.asarray(sgI[c], float) ** 2, 0.01) for c in pool}
    st_rows, rate_rows, unc_rows, wy_unc = {}, [], [], []
    loo_series, plane_series, plane_joint_series = {}, {}, {}
    for k in series_ids:
        cal = [c for c in pool if c != k]
        o, consts, info = _loo_datum(cal, Y, Wt, n_t)
        A = I[k] + o
        loo_series[k] = A
        po = _plane_datum(cal, Y, consts, xy, xy[k], n_t) if consts else np.full(n_t, np.nan)
        plane_series[k] = I[k] + po
        pj, why_pj = _joint_plane_datum(cal, Y, Wt, xy, xy[k], sar_days)
        plane_joint_series[k] = I[k] + pj
        m = np.isfinite(A) & np.isfinite(G[k])
        row = {"datum": info, "epochs": int(m.sum())}
        if m.sum() >= 3:
            diff = (A - G[k])[m]
            off = float(diff.mean())
            res = diff - off
            row.update(offset_mm=_f(off, 2), rmse=float(np.sqrt(np.mean(res ** 2))), r=pearson(A[m], G[k][m]),
                       first=_iso(sar_days[m][0]), last=_iso(sar_days[m][-1]))
            if sig_range and k in sig_range:
                s, rng_ = sig_range[k]
                row["signal_range_mm"], row["daily_sigma_mm"] = _f(rng_, 2), _f(s, 2)
                row["qualifies_for_correlation"] = bool(math.isfinite(s) and math.isfinite(rng_) and rng_ > 10 * s)
        mp = np.isfinite(plane_series[k]) & np.isfinite(G[k])
        if mp.sum() >= 3:
            dp = (plane_series[k] - G[k])[mp]
            row["plane_rmse"] = float(np.sqrt(np.mean((dp - dp.mean()) ** 2)))
            row["plane_epochs"] = int(mp.sum())
        mj = np.isfinite(plane_joint_series[k]) & np.isfinite(G[k])
        if mj.sum() >= 3:
            dj = (plane_joint_series[k] - G[k])[mj]
            row["plane_joint_rmse"] = float(np.sqrt(np.mean((dj - dj.mean()) ** 2)))
            row["plane_joint_epochs"] = int(mj.sum())
        elif why_pj:
            row["plane_joint_why"] = why_pj
        if k in rate_ids and m.sum() >= 4:
            t = sar_days[m]
            rA, rG = model_rates(t, A[m], prm), model_rates(t, G[k][m], prm)
            for lab in rA["rates"]:
                if lab in rG["rates"]:
                    sI = max(rA["rates"][lab]["sigma"], stated.get(k) or 0.0)
                    rate_rows.append({"station": k, "rate": lab, "insar": rA["rates"][lab]["rate"],
                                      "gnss": rG["rates"][lab]["rate"], "s_insar": sI,
                                      "s_gnss": rG["rates"][lab]["sigma"],
                                      "s_stated": stated.get(k)})
                    if stated.get(k) is not None:
                        e = rA["rates"][lab]["rate"] - rG["rates"][lab]["rate"]
                        lim = 1.96 * math.hypot(stated[k], rG["rates"][lab]["sigma"])
                        wy_unc.append({"station": k, "rate": lab, "error": e, "limit": lim, "inside": abs(e) <= lim})
            src_A = rA["overlap_mean"] if uc_key == "overlap_mean" else next(iter(rA["rates"].values()), None)
            src_G = rG["overlap_mean"] if uc_key == "overlap_mean" else next(iter(rG["rates"].values()), None)
            if src_A and src_G:
                e = src_A["rate"] - src_G["rate"]
                s_st = stated.get(k)
                lim = 1.96 * math.hypot(s_st, src_G["sigma"]) if s_st is not None else None
                unc_rows.append({"station": k, "insar": src_A["rate"], "gnss": src_G["rate"], "error": e,
                                 "s_stated": s_st, "s_gnss": src_G["sigma"], "limit": lim,
                                 "inside": (abs(e) <= lim) if lim is not None else None})
            row["rates"] = {"insar": {lb: _f(v["rate"], 2) for lb, v in rA["rates"].items()},
                            "gnss": {lb: _f(v["rate"], 2) for lb, v in rG["rates"].items()},
                            "overlap_mean": {"insar": _f((rA["overlap_mean"] or {}).get("rate"), 2),
                                             "gnss": _f((rG["overlap_mean"] or {}).get("rate"), 2)}}
        st_rows[k] = row
    out["loo"] = {"stations": st_rows, "series": loo_series, "plane_series": plane_series,
                  "plane_joint_series": plane_joint_series,
                  "plane_note": ("plane_rmse: the protocol's offset_plus_plane (sensitivity, never graded). "
                                 "plane_joint_rmse: the same plane fitted jointly with the station constants "
                                 "(significance.gnss_datum_fit), not in the protocol and not graded; the "
                                 "protocol's two-stage plane is biased when stations join or leave")}
    if not any("rmse" in v for v in st_rows.values()):
        reasons = sorted({v["datum"].get("why") or f"{v['epochs']} matched epochs" for v in st_rows.values()})
        why["loo"] = ("no station could be scored leave-one-out: " + "; ".join(reasons)) if reasons else \
            "no station to score"
    elif not rate_rows:
        why["loo_rates"] = "no scored station has a rate over its matched record"
    out["why"] = why

    # ---------------- rate rows (pooled station x rate)
    by_st = {}
    for r_ in rate_rows:
        by_st.setdefault(r_["station"], []).append(r_)
    st_list = list(by_st)
    x = np.array([r_["gnss"] for r_ in rate_rows])
    y = np.array([r_["insar"] for r_ in rate_rows])
    sx = np.array([r_["s_gnss"] for r_ in rate_rows])
    sy = np.array([r_["s_insar"] for r_ in rate_rows])
    err = y - x if x.size else np.array([])

    def pooled(idx):
        rows = [r_ for i in idx for r_ in by_st[st_list[i]]]
        return (np.array([r_["gnss"] for r_ in rows]), np.array([r_["insar"] for r_ in rows]),
                np.array([r_["s_gnss"] for r_ in rows]), np.array([r_["s_insar"] for r_ in rows]))

    def boot_bias(idx, _r):
        gx, gy, _, _ = pooled(idx)
        return float(np.mean(gy - gx)) if gx.size else np.nan

    def boot_slope(idx, _r):
        f = york_fit(*pooled(idx))
        return f["slope"] if f else np.nan

    def boot_r(idx, _r):
        gx, gy, _, _ = pooled(idx)
        return pearson(gx, gy)

    es = error_stats(err)
    n_st = len(st_list)
    out["rates"] = {"rows": [{k: (_f(v, 3) if isinstance(v, float) else v) for k, v in r_.items()} for r_ in rate_rows],
                    "stats": es,
                    "ci_bias": bootstrap_ci(n_st, boot_bias, n_boot, seed) if es.get("n") else {"lo": None, "hi": None}}
    yf = york_fit(x, y, sx, sy) if x.size else None
    out["york"] = {"fit": yf, "ci_slope": bootstrap_ci(n_st, boot_slope, n_boot, seed) if yf else {"lo": None, "hi": None}}
    r_all = pearson(x, y)
    out["correlation"] = {"r": r_all, "r2": r_all ** 2 if math.isfinite(r_all) else float("nan"), "n": int(x.size),
                          "ci_r": bootstrap_ci(n_st, boot_r, n_boot, seed) if math.isfinite(r_all) else {"lo": None, "hi": None}}
    # The leave-one-out datum carries the whole box's motion from the calibration GNSS into the InSAR,
    # so water-year rates pooled across years correlate, and line up on a slope near 1, through that
    # common motion alone (an InSAR reading 30% of the relative motion low still gives a York slope
    # near 1 at Corcoran). The same two statistics with each water year's mean over the stations
    # removed are reported beside the graded ones: never graded, since the protocol fixes the rows.
    if wy and rate_rows:
        xd, yd, sxd, syd = [], [], [], []
        for lab in sorted({r_["rate"] for r_ in rate_rows}):
            rr = [r_ for r_ in rate_rows if r_["rate"] == lab]
            if len(rr) < 2:
                continue
            gx, gy = np.array([r_["gnss"] for r_ in rr]), np.array([r_["insar"] for r_ in rr])
            xd += list(gx - gx.mean())
            yd += list(gy - gy.mean())
            sxd += [r_["s_gnss"] for r_ in rr]
            syd += [r_["s_insar"] for r_ in rr]
        yfd = york_fit(xd, yd, sxd, syd) if len(xd) >= 3 else None
        out["common_mode_removed"] = {
            "what": "the same rates with each water year's mean over the stations removed (reported, not graded)",
            "n": len(xd), "r": _f(pearson(xd, yd), 3), "york_slope": _f(yfd["slope"], 3) if yfd else None,
            "york_slope_se": _f(yfd["slope_se"], 3) if yfd else None}

    # ---------------- series rows (per station; graded on the median)
    rm = {k: v["rmse"] for k, v in st_rows.items() if "rmse" in v}
    rk = list(rm)
    rv = np.array([rm[k] for k in rk])
    out["series_rmse"] = {"per_station": {k: _f(v, 2) for k, v in rm.items()},
                          "median": float(np.median(rv)) if rv.size else float("nan"),
                          "ci_median": bootstrap_ci(len(rk), lambda idx, _r: float(np.median(rv[idx])), n_boot, seed)
                          if rv.size else {"lo": None, "hi": None}}
    qual = {k: v["r"] for k, v in st_rows.items() if v.get("qualifies_for_correlation") and math.isfinite(v.get("r", np.nan))}
    qk = list(qual)
    qv = np.array([qual[k] for k in qk])
    out["series_correlation"] = {"qualifying": qk, "per_station": {k: _f(v, 3) for k, v in qual.items()},
                                 "median": float(np.median(qv)) if qv.size else float("nan"),
                                 "ci_median": bootstrap_ci(len(qk), lambda idx, _r: float(np.median(qv[idx])), n_boot, seed)
                                 if qv.size else {"lo": None, "hi": None},
                                 "not_qualifying": [k for k, v in st_rows.items() if "rmse" in v and not v.get("qualifies_for_correlation")]}

    # ---------------- uncertainty calibration
    known = [u for u in unc_rows if u["inside"] is not None]
    n_in = sum(1 for u in known if u["inside"])
    out["uncertainty"] = {"rate_used": uc_key, "rows": [{k: (_f(v, 3) if isinstance(v, float) else v) for k, v in u.items()}
                                                        for u in unc_rows],
                          "n": len(known), "inside": n_in, "outside": len(known) - n_in,
                          "share_inside": n_in / len(known) if known else float("nan"),
                          "also_station_x_rate": {"n": len(wy_unc), "inside": sum(1 for u in wy_unc if u["inside"]),
                                                  "share_inside": (sum(1 for u in wy_unc if u["inside"]) / len(wy_unc))
                                                  if wy_unc else None}}
    if not wy and not rate_rows:
        out["note"] = "no station had a rate (too short a matched record)"

    # ---------------- amendments of 8 Oct 2026 (before any result)
    amend = prm.get("amend") or {}
    if amend.get("york_slope_on") == "interval_changes" and prm.get("change_intervals"):
        ch_ids = list(dict.fromkeys(series_ids + rate_ids))
        s_ep = {s: _insar_epoch_sigma(sar_days, I[s], prm) for s in ch_ids}
        out["changes"] = change_slope(interval_changes(sar_days, I, G, sgG, prm, ch_ids), s_ep, n_boot, seed)
    if amend.get("plane_check") == "joint_graded":
        out["plane_check"] = _datum_scores(plane_joint_series, G, sar_days, rate_ids, series_ids, stated, prm, P,
                                           sig_range, n_boot, seed)
        out["plane_check"]["what"] = ("leave-one-out rows in the joint offset + plane datum "
                                      "(significance.gnss_datum_fit, plane_method 'joint'; amendment 2026-10-08/3)")
    return out


def _network_rows(comp: dict, P: dict) -> dict:
    """Metrics per graded row, from network_component's output, graded
    against the protocol's thresholds."""
    C = P.get("comparisons") or {}
    rows = {}

    why = comp.get("why") or {}
    reason = {"rate_rmse_dd": why.get("dd"), "series_rmse": why.get("loo"), "series_correlation": why.get("loo")}
    for r_ in ("rate_bias_loo", "york_slope", "rate_correlation", "uncertainty_calibration"):
        reason[r_] = why.get("loo") or why.get("loo_rates")

    def put(name, M, extra):
        spec = (C.get(name) or {}).get("grade")
        g = grade_row(spec, M) if name in C else _labelled("not_computed", "the protocol has no such row")
        if g["grade"] == "not_testable" and reason.get(name) and "why" not in g:
            g["why"] = reason[name]
        rows[name] = {"what": (C.get(name) or {}).get("what"), **g, "metrics": M, **extra}

    dd = comp["dd"]
    put("rate_rmse_dd", {"rmse": dd["stats"].get("rmse"), "n": dd["stats"].get("n")},
        {"ci": {"rmse": dd["ci_rmse"]}, "stats": dd["stats"], "pairs": dd["pairs"]})
    rs = comp["rates"]["stats"]
    bias = rs.get("bias")
    put("rate_bias_loo", {"abs_bias": abs(bias) if bias is not None else None, "bias": bias, "n": rs.get("n")},
        {"ci": {"bias": comp["rates"]["ci_bias"]}, "stats": rs})
    yf = comp["york"]["fit"]
    ci = comp["york"]["ci_slope"]
    inc1 = (ci["lo"] <= 1.0 <= ci["hi"]) if ci.get("lo") is not None else None
    cm = {"common_mode_removed": comp["common_mode_removed"]} if comp.get("common_mode_removed") else {}
    put("york_slope", {"slope": yf["slope"] if yf else None, "ci_includes_1": inc1},
        {"ci": {"slope": ci}, "fit": yf, **cm})
    cr = comp["correlation"]
    put("rate_correlation", {"r": cr["r"] if math.isfinite(cr["r"]) else None, "r2": cr["r2"], "n": cr["n"]},
        {"ci": {"r": cr["ci_r"]}, **cm})
    sr = comp["series_rmse"]
    put("series_rmse", {"median_rmse": sr["median"] if math.isfinite(sr["median"]) else None},
        {"ci": {"median_rmse": sr["ci_median"]}, "per_station": sr["per_station"],
         "per_station_grade": {k: grade_row((C.get("series_rmse") or {}).get("grade"), {"median_rmse": v})["label"]
                               for k, v in sr["per_station"].items() if v is not None}})
    sc = comp["series_correlation"]
    put("series_correlation", {"r": sc["median"] if math.isfinite(sc["median"]) else None, "n_qualifying": len(sc["qualifying"])},
        {"ci": {"r": sc["ci_median"]}, "per_station": sc["per_station"], "not_qualifying": sc["not_qualifying"]})
    if not sc["qualifying"] and "series_correlation" in rows:
        rows["series_correlation"].update(_labelled(
            "not_testable", why.get("loo") or "no station's GNSS signal exceeds 10 x its daily noise"))
    uc = comp["uncertainty"]
    put("uncertainty_calibration", {"share_inside": uc["share_inside"] if uc["n"] else None,
                                    "n_outside": uc["outside"] if uc["n"] else None, "n": uc["n"]},
        {"rows": uc["rows"], "rate_used": uc["rate_used"], "also_station_x_rate": uc["also_station_x_rate"]})
    _amended_network_rows(rows, comp, P, put)
    return rows


def _amended_network_rows(rows: dict, comp: dict, P: dict, put) -> None:
    """The rows the amendments of 8 Oct 2026 change (protocol B1, B2):
    2026-10-08/2 grades york_slope on the per-station interval changes and
    keeps the rate-based slope and rate_correlation as reported, not graded;
    2026-10-08/3 adds the graded plane_check (joint offset + plane datum)."""
    sw = amendment_switches(P)
    C = P.get("comparisons") or {}
    ch = comp.get("changes")
    if sw.get("york_slope_on") == "interval_changes" and "york_slope" in rows:
        old = rows["york_slope"]
        rate_based = {"slope": old["metrics"].get("slope"), "ci_includes_1": old["metrics"].get("ci_includes_1"),
                      "ci": old.get("ci"), "fit": old.get("fit"),
                      "note": "rate-based York slope as first written: reported, not graded (amendment 2026-10-08/2)"}
        if ch is None:
            M = {"slope": None, "ci_includes_1": None, "n": 0}
            put("york_slope", M, {"quantity": "per-station interval changes (amendment 2026-10-08/2)",
                                  "rate_based": rate_based})
            rows["york_slope"].update(_labelled("not_testable", "no change interval in this run's window"))
        else:
            fit = ch.get("fit")
            M = {"slope": fit["slope"] if fit else None, "ci_includes_1": ch.get("ci_includes_1"), "n": ch["n"]}
            put("york_slope", M, {"quantity": ch["what"], "ci": {"slope": ch["ci_slope"]}, "fit": fit,
                                  "changes_r": _f(ch.get("r"), 4), "intervals": ch["intervals"],
                                  "insar_epoch_sigma_mm": ch["insar_epoch_sigma_mm"], "rate_based": rate_based,
                                  **({"common_mode_removed": old["common_mode_removed"]}
                                     if old.get("common_mode_removed") else {})})
            if rows["york_slope"]["grade"] == "not_testable" and ch.get("why"):
                rows["york_slope"]["why"] = ch["why"]
        rows["york_slope"]["amendment"] = "2026-10-08/2"
    if sw.get("rate_slope_and_correlation") == "report_only" and "rate_correlation" in rows:
        r = rows["rate_correlation"]
        r.update(_labelled("report_only", "rate-based: reported, not graded (amendment 2026-10-08/2)"),
                 amendment="2026-10-08/2")
        r.pop("missing", None)
        if ch is not None:
            r["changes_r"] = _f(ch.get("r"), 4)
    pc = comp.get("plane_check")
    if sw.get("plane_check") == "joint_graded" and pc is not None:
        sub = {}
        for name in ("rate_bias_loo", "series_rmse", "series_correlation", "uncertainty_calibration"):
            M = {k: v for k, v in pc[name].items() if not k.startswith("ci") and k not in ("per_station", "rate_used")}
            g = grade_row((C.get(name) or {}).get("grade"), M) if name in C else _labelled("not_computed", "no such row")
            if name == "series_correlation" and not pc[name]["n_qualifying"]:
                g = _labelled("not_testable", "no station's GNSS signal exceeds 10 x its daily noise, or no value")
            sub[name] = {**g, "metrics": M, **{k: v for k, v in pc[name].items() if k.startswith("ci")}}
        grades = [s["grade"] for s in sub.values()]
        if "fail" in grades:
            top = "fail"
        elif all(x == "pass" for x in grades):
            top = "pass"
        elif any(x in ("not_testable", "not_computed") for x in grades):
            top = "not_testable"
        else:
            top = "marginal"
        rows["plane_check"] = {"what": pc["what"], **_labelled(top), "rows": sub, "per_station": pc["per_station"],
                               "amendment": "2026-10-08/3",
                               "note": "graded against the offset-only rows' thresholds; reported beside them; "
                                       "not part of the route verdict (offset-only is primary)"}


def run_network(P, prm, prod, gnss, samples, floors, up=None) -> dict:
    """Network design (protocols B1, B2): every row at the primary floor,
    the LOS rows again at each sensitivity floor, and the secondary vertical
    rows (LOS / cos(incidence), and D1 Up when given)."""
    sar = prod.days
    half = prm["match_days"]
    sts = [s for s in prm["stations"] if s in gnss and s in samples]
    xy = _local_xy_km({s: prm["stations"][s] for s in sts})
    result = {"design": "network", "stations_used": sts, "by_floor": {}, "epoch_table": []}
    matched = {s: match_epochs(gnss[s], sar, half) for s in sts}
    for fl in floors:
        I, G, sgI, sgG, stated, sr = {}, {}, {}, {}, {}, {}
        Iv, Gv, sgIv, sgGv, statedv, srv = {}, {}, {}, {}, {}, {}
        for s in sts:
            smp = samples[s][fl]
            l = smp["los"]
            if not np.all(np.isfinite(l)) or not np.isfinite(smp["series"]).any():
                continue
            enu, nd = matched[s]
            I[s], G[s] = smp["series"], enu @ l
            sg, rg = daily_sigma_and_range(gnss[s], l, prm)
            sgI[s], sgG[s], stated[s], sr[s] = smp["err"], np.full(sar.size, sg), smp["stated_rate_sigma"], (sg, rg)
            lu = l[2]
            Iv[s], Gv[s] = smp["series"] / lu, enu[:, 2]
            sgv, rgv = daily_sigma_and_range(gnss[s], np.array([0.0, 0.0, 1.0]), prm)
            sgIv[s], sgGv[s], srv[s] = smp["err"] / lu, np.full(sar.size, sgv), (sgv, rgv)
            statedv[s] = smp["stated_rate_sigma"] / lu if smp["stated_rate_sigma"] is not None else None
        comp = network_component(sar, I, G, sgI, sgG, stated, prm, P, xy, sr)
        entry = {"los": comp, "rows": _network_rows(comp, P), "_G": G}
        if fl == prm["floor"]:
            vcomp = network_component(sar, Iv, Gv, sgIv, sgGv, statedv, prm, P, xy, srv)
            entry["vertical_assumed"] = vcomp
            entry["vertical_assumed_rows"] = _network_rows(vcomp, P)
            if up is not None:
                # D1 Up on its own grid and epochs; GNSS Up matched at those epochs
                ud = up["days"]
                Iu, Gu, sgIu, sgGu = {}, {}, {}, {}
                for s in Iv:
                    u = up["stations"].get(s)
                    if u is None or not np.isfinite(u["series"]).any():
                        continue
                    Iu[s], sgIu[s] = u["series"], u["err"]
                    Gu[s] = match_epochs(gnss[s], ud, half)[0][:, 2]
                    sgGu[s] = np.full(ud.size, srv[s][0])
                ucomp = network_component(ud, Iu, Gu, sgIu, sgGu, {s: None for s in Iu}, prm, P, xy,
                                          {s: srv[s] for s in Iu}) if Iu else None
                entry["up_d1"] = ucomp
                entry["up_d1_rows"] = _network_rows(ucomp, P) if ucomp else None
            for s in I:
                enu, nd = matched[s]
                smp = samples[s][fl]
                A = comp["loo"]["series"].get(s)
                for i, d in enumerate(prod.dates):
                    if not (np.isfinite(I[s][i]) or np.isfinite(G[s][i])):
                        continue
                    result["epoch_table"].append({
                        "station": s, "date": d, "insar_los_mm": _f(I[s][i], 2), "gnss_los_mm": _f(G[s][i], 2),
                        "gnss_days": int(nd[i]), "loo_absolute_insar_los_mm": _f(A[i], 2) if A is not None else None,
                        "n_cells": int(smp["n_cells"][i]), "n_independent": int(smp["n_independent"][i]),
                        "insar_spread_mm": _f(smp["spread"][i], 2), "insar_iqr_mm": _f(smp["iqr"][i], 2),
                        "gnss_e_mm": _f(enu[i, 0], 2), "gnss_n_mm": _f(enu[i, 1], 2), "gnss_u_mm": _f(enu[i, 2], 2)})
            # plate motion: the part of each pair's GNSS double difference that comes from the LOS
            # vector differing between the stations, (l_i - l_j) . ENU_j - computed and reported, not removed
            pm = []
            for p in comp["dd"]["pairs"]:
                i, j = p["i"], p["j"]
                term = (samples[i][fl]["los"] - samples[j][fl]["los"]) @ matched[j][0].T
                m = np.isfinite(term)
                f = fit_series(sar[m], term[m], (), False) if m.sum() >= 4 else None
                pm.append({"pair": p["pair"], "rate_mm_yr": _f(f["coef"][1], 3) if f else None})
            entry["los_vector_variation"] = {
                "what": "rate of (l_i - l_j) . ENU_j: the part of a pair's GNSS LOS difference due to the LOS "
                        "vector varying across the box (plate motion and common motion leak through it)",
                "pairs": pm}
        result["by_floor"][fl] = entry
    return result


# =============================================================== stable ground (protocol A, amendment 2026-10-08/1)

_DRAW_KEYS = ("draw_distance_m", "draw_m")
_DEPTH_KEYS = ("depth_of_cover_m", "cover_depth_m", "depth_m", "depth")
DEFAULT_DRAW_M = 300.0              # protocol A site.angle_of_draw: "300 m where the depth is not stated"


def _isnum(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _grid_lalo(attrs, rows, cols):
    """Cell-centre (lat, lon) of rows/cols on a geographic or projected MintPy grid."""
    from . import deliverables as dlv
    x0, dx, y0, dy = dlv._geotransform(attrs)
    x = x0 + (np.asarray(cols, float) + 0.5) * dx
    y = y0 + (np.asarray(rows, float) + 0.5) * dy
    crs = dlv._crs(attrs)
    if crs == "EPSG:4326":
        return y, x
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    return np.asarray(lat, float), np.asarray(lon, float)


def _km_between(lat0, lon0, lat, lon):
    """Distance in km (equirectangular: exact enough over a 20-30 km box)."""
    k = math.cos(math.radians(float(lat0)))
    return np.hypot((np.asarray(lon, float) - float(lon0)) * 111.32 * k, (np.asarray(lat, float) - float(lat0)) * 110.57)


def stable_ground_mask(prod, cfg, outlines=None, ring_km=(1.0, 3.0)) -> tuple[np.ndarray | None, str]:
    """The stable ground of amendment 2026-10-08/1, on the product grid:
    the stable ring of protocol A (cells ring_km[0]-ring_km[1] km outside every
    T1 panel outline + its draw distance: a feature's draw_distance_m, else
    0.5 x its depth of cover, else 300 m) when the outlines are given; else
    the reference area (cells more than the run's deramp_buffer_m, default
    2 km, from its deramp_exclude_geojson). Only imaged cells (finite temporal
    coherence: the processed crop). (mask or None, how it was made)."""
    import tempfile
    from .stack_backend import _polygon_mask
    sg = _significance()
    attrs = {**prod.attrs, "LENGTH": str(prod.shape[0]), "WIDTH": str(prod.shape[1])}
    imaged = np.isfinite(prod.tcoh)
    if outlines and Path(outlines).exists():
        gj = json.loads(Path(outlines).read_text())
        feats = [f for f in gj.get("features", []) or []
                 if ((f or {}).get("geometry") or {}).get("type") in ("Polygon", "MultiPolygon")]
        if not feats:
            return None, f"{outlines} holds no polygon"
        groups: dict = {}
        for f in feats:
            p = f.get("properties") or {}
            draw = next((float(p[k]) for k in _DRAW_KEYS if _isnum(p.get(k))), None)
            if draw is None:
                dep = next((float(p[k]) for k in _DEPTH_KEYS if _isnum(p.get(k))), None)
                draw = 0.5 * dep if dep else DEFAULT_DRAW_M
            groups.setdefault(round(draw, 1), []).append(f)
        panels = np.zeros(prod.shape, bool)
        with tempfile.TemporaryDirectory() as td:
            for k, (draw, fs) in enumerate(sorted(groups.items())):
                gp = Path(td) / f"draw_{k}.geojson"
                gp.write_text(json.dumps({"type": "FeatureCollection", "features": fs}))
                m = np.asarray(_polygon_mask(gp, prod.shape, attrs), bool)
                panels |= sg.euclidean_buffer(m, attrs, draw) if m.any() else m
        if not panels.any():
            return None, f"no panel outline of {outlines} falls on the product grid"
        ring = sg.euclidean_buffer(panels, attrs, ring_km[1] * 1000.0) & ~sg.euclidean_buffer(panels, attrs, ring_km[0] * 1000.0)
        draws = ", ".join(f"{d:g} m ({len(fs)} outline(s))" for d, fs in sorted(groups.items()))
        return ring & imaged, (f"stable ring {ring_km[0]:g}-{ring_km[1]:g} km outside the T1 panel outlines + draw "
                               f"distance ({outlines}; draw distances {draws})")
    dp = str(getattr(cfg, "deramp_exclude_geojson", "") or "").strip()
    if dp and Path(dp).exists():
        buf = float(getattr(cfg, "deramp_buffer_m", 0) or getattr(cfg, "acceptance_far_field_buffer_m", 0) or 2000.0)
        m = np.asarray(_polygon_mask(Path(dp), prod.shape, attrs), bool)
        if not m.any():
            return None, f"the deramp polygon {dp} covers no cell of the product grid"
        return ~sg.euclidean_buffer(m, attrs, buf) & imaged, (f"reference area: cells more than {buf:g} m from the "
                                                             f"run's deramp polygon {dp} (no T1 outlines given)")
    return None, "no stable ground: no T1 panel outlines given and no deramp_exclude_geojson in the run's config"


def sample_placements(prod, lat, lon, radius_m: float, floor: float, min_cells: int, min_indep: int) -> dict:
    """The protocol's sampling (median of reliable cells within the radius,
    minimum cells and independent radar pixels) at many disc centres at once,
    at one coherence floor: one read of the cube for all of them.
    {'series': (n_dates, n) mm, 'los_up': (n,), 'cells': (n,)}"""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    n_t, n_q = len(prod.dates), lat.size
    series, lu_out, ncell = np.full((n_t, n_q), np.nan), np.full(n_q, np.nan), np.zeros(n_q, int)
    parts = [prod.cells_near(la, lo, radius_m) for la, lo in zip(lat, lon)]
    if not n_q or not any(p[0].size for p in parts):
        return {"series": series, "los_up": lu_out, "cells": ncell}
    R = np.concatenate([p[0] for p in parts])
    C = np.concatenate([p[1] for p in parts])
    owner = np.concatenate([np.full(p[0].size, k) for k, p in enumerate(parts)])
    tc = prod.tcoh[R, C]
    rel = np.isfinite(tc) & (tc >= floor)
    keep = rel
    R, C, owner = R[keep], C[keep], owner[keep]
    if not R.size:
        return {"series": series, "los_up": lu_out, "cells": ncell}
    reliable = {"rows": R, "cols": C, "owner": owner}              # each disc's reliable cells (the flag test)
    ts = prod.read_cells(R, C)
    ids, _ = prod.pixel_ids(R, C)
    geo = [prod.geometry(k) for k in ("los_east", "los_north", "los_up")]
    order = np.argsort(owner, kind="stable")
    bounds = np.searchsorted(owner[order], np.arange(n_q + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(n_q):
            sel = order[bounds[k]:bounds[k + 1]]
            if not sel.size:
                continue
            ncell[k] = sel.size
            v = ts[:, sel]
            fin = np.isfinite(v)
            pid = ids[sel]
            n_c = fin.sum(1)
            n_i = np.array([np.unique(pid[fin[i]]).size for i in range(n_t)])
            med = np.nanmedian(v, axis=1)
            series[:, k] = np.where((n_c >= min_cells) & (n_i >= min_indep), med, np.nan)
            if all(g is not None for g in geo):
                vec = np.array([np.nanmedian(g[R[sel], C[sel]]) for g in geo])
                nrm = float(np.linalg.norm(vec)) if np.all(np.isfinite(vec)) else float("nan")
                lu_out[k] = vec[2] / nrm if nrm and math.isfinite(nrm) else np.nan
    return {"series": series, "los_up": lu_out, "cells": ncell, "reliable": reliable}


def stable_placements(prod, cfg, prm, avoid_lalo, outlines=None) -> dict:
    """Disc centres on the stable ground (stable_ground_mask) on a square grid
    of spacing max(2 x radius, sqrt(area / stable_max_placements)), so the
    discs never overlap and there are at most about that many; discs within
    2 x radius of the reference station or a truth point (avoid_lalo) are
    left out. Sampled with the protocol's rules at the primary floor."""
    sw = prm.get("amend") or {}
    ring = tuple(float(x) for x in (sw.get("stable_ring_km") or (1.0, 3.0)))
    mask, how = stable_ground_mask(prod, cfg, outlines, ring)
    if mask is None or not mask.any():
        return {"why": how if mask is None else f"{how}: no imaged cell in it", "how": how}
    sg = _significance()
    dy_m, dx_m = sg.pixel_metres({**prod.attrs, "LENGTH": str(prod.shape[0])})
    area = float(mask.sum()) * dy_m * dx_m
    r = float(prm["radius_m"])
    nmax = max(1, int(sw.get("stable_max_placements", 300)))
    spacing = max(2.0 * r, math.sqrt(area / nmax))
    sr, sc = max(1, int(round(spacing / dy_m))), max(1, int(round(spacing / dx_m)))
    rr, cc = np.meshgrid(np.arange(sr // 2, prod.shape[0], sr), np.arange(sc // 2, prod.shape[1], sc), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    on = mask[rr, cc]
    rr, cc = rr[on], cc[on]
    lat, lon = _grid_lalo(prod.attrs, rr, cc)
    far = np.ones(lat.size, bool)
    for la, lo in avoid_lalo:
        far &= _km_between(la, lo, lat, lon) * 1000.0 > 2.0 * r
    lat, lon = lat[far], lon[far]
    s = sample_placements(prod, lat, lon, r, prm["floor"], prm["min_cells"], prm["min_independent_pixels"])
    have = np.isfinite(s["series"]).any(0)
    return {"how": how, "lat": lat, "lon": lon, "series": s["series"], "los_up": s["los_up"],
            "_cells": s.get("reliable"),
            "n_placements": int(lat.size), "n_with_a_value": int(have.sum()), "spacing_m": round(spacing, 1),
            "stable_area_km2": round(area / 1e6, 2), "radius_m": r, "floor": prm["floor"]}


def stable_change_sigma(d, dist_km, p_km, sw) -> dict:
    """The InSAR change's 1-sigma from the same double-differenced change at
    the stable placements (amendment 2026-10-08/1): 1.4826 x median |d| about
    ZERO (stable ground should not move; a shift of all of it is error too),
    over the placements whose distance to the reference station is 0.5-2 x
    the point's own; all placements when fewer than the minimum lie in that
    band; 'not stated' when fewer than the minimum have a value."""
    lo, hi = (float(x) for x in (sw.get("stable_distance_band") or (0.5, 2.0)))
    nmin = int(sw.get("stable_min_placements", 30))
    d = np.asarray(d, float)
    dist_km = np.asarray(dist_km, float)
    ok = np.isfinite(d)
    pk = float(p_km) if p_km is not None and math.isfinite(float(p_km)) and float(p_km) > 0 else None
    band = (ok & (dist_km >= lo * pk) & (dist_km <= hi * pk)) if pk else np.zeros_like(ok)
    n_band = int(band.sum())
    used = (f"placements {lo:g}-{hi:g} x the point's distance to the reference ({lo * pk:.1f}-{hi * pk:.1f} km)"
            if pk else "")
    if n_band < nmin:
        band, used = ok, (f"all placements: {n_band} with a value lie {lo:g}-{hi:g} x the point's distance to the "
                          f"reference, fewer than {nmin}")
    if band.sum() < nmin:
        return {"sigma_mm": None, "n": int(band.sum()),
                "why": f"{int(band.sum())} stable placements have a value at both dates; the interval needs {nmin}"}
    return {"sigma_mm": float(1.4826 * np.median(np.abs(d[band]))), "n": int(band.sum()), "placements": used,
            "median_distance_km": _f(np.median(dist_km[band]), 2), "point_distance_km": _f(pk, 2),
            "median_change_mm": _f(np.median(d[band]), 2)}


def _interval_fields(rec: dict, s_insar: dict, s_truth) -> None:
    """Write the amended 95% interval of a comparison into its record."""
    rec["insar_sigma"] = {k: (_f(v, 2) if isinstance(v, float) else v) for k, v in s_insar.items()}
    si = s_insar.get("sigma_mm")
    if si is None or s_truth is None or not math.isfinite(float(s_truth)):
        rec["interval_95_mm"] = None
        rec["interval_why"] = s_insar.get("why") or "no truth sigma"
        return
    lim = 1.96 * math.hypot(si, float(s_truth))
    rec["interval_95_mm"] = _f(lim, 1)
    rec["interval_truth_sigma_mm"] = _f(s_truth, 2)
    if rec.get("error") is not None:
        rec["inside_interval"] = bool(abs(float(rec["error"])) <= lim)


def _interp_cols(S, sar, day, gap):
    """Rows of S (n_dates, n) interpolated to `day` by the survey-epoch rule:
    the SAR date itself, else linear between neighbours at most `gap` days
    apart; NaN outside the dates or across a longer gap (never filled)."""
    j = int(np.searchsorted(sar, day))
    if j < sar.size and sar[j] == day:
        return S[j].astype(float)
    if j == 0 or j >= sar.size or sar[j] - sar[j - 1] > gap:
        return np.full(S.shape[1], np.nan)
    a, b = j - 1, j
    return S[a] + (S[b] - S[a]) * (day - sar[a]) / (sar[b] - sar[a])


# =============================================================== reference-station design (protocol A)

def _event_dates(text):
    m0 = re.search(r"t0\s*=\s*last\s+\S+\s+date\s*<=\s*(" + DATE_RE + ")", str(text or ""))
    m1 = re.search(r"t1\s*=\s*first\s+\S+\s+date\s*>=\s*(" + DATE_RE + ")", str(text or ""))
    return (m0.group(1), m1.group(1)) if m0 and m1 else None


def _series_span(text):
    m = re.search(r"every\s+\S+\s+date\s+in\s+(" + DATE_RE + r")\s*\.\.\s*(" + DATE_RE + ")", str(text or ""))
    return (m.group(1), m.group(2)) if m else None


def _not_this_run(c, run):
    """Why a protocol row does not apply to this run (its `runs:` list names
    other runs), else None. Without a known run every row applies."""
    rl = c.get("runs") if isinstance(c, dict) else None
    if run and isinstance(rl, list) and rl and str(run) not in [str(x) for x in rl]:
        return f"the protocol runs this row on {', '.join(str(x) for x in rl)}, not on {run}"
    return None


def _flags_for(prod, samples, fl, flag_layer):
    """{point: True/False/None}: flagged when at least half of the reliable
    cells it was sampled from carry the flag."""
    if flag_layer is None:
        return {}
    out = {}
    for pid, per in samples.items():
        if pid.startswith("_"):
            continue
        smp = per[fl]
        if smp["rows"].size:
            v = flag_layer[smp["rows"], smp["cols"]]
            v = v[np.isfinite(v)]
            out[pid] = bool(v.size and np.mean(v > 0) >= 0.5)
    return out


def read_flag_layer(cfg, shape) -> tuple[np.ndarray | None, str, bool]:
    """(flag grid, what it is, whether it may grade). F2's
    quality_aliasing_risk.tif grades; until F2 exists the old
    quality_too_fast_to_measure.tif gives a labelled lower bound that does
    not grade (protocol A, unflagged_failure_rate)."""
    d = Path(cfg.project_dir) / "deliverables"
    try:
        from . import deliverables as dlv
        ts_mtime = Path(dlv.timeseries_file(cfg)).stat().st_mtime
    except Exception:                                     # noqa: BLE001 - no age check then
        ts_mtime = None
    for name, grades, what in ((ALIAS_TIF, True, "F2 aliasing-risk flag (whole record: every pair)"),
                               ("quality_too_fast_to_measure.tif", False,
                                "old quality_too_fast_to_measure flag: lower bound, unwrapped velocity")):
        hits = sorted(d.rglob(name)) if d.exists() else []
        if hits and ts_mtime is not None and hits[0].stat().st_mtime < ts_mtime:
            log.warning("%s is older than the time series (an earlier inversion's): not used", hits[0])
            continue
        if hits:
            try:
                import rasterio
                with rasterio.open(hits[0]) as src:
                    a = src.read(1).astype(float)
                    if src.nodata is not None:
                        a[a == src.nodata] = np.nan
                if name == ALIAS_TIF:
                    a[(a < 0) | (a > 2)] = np.nan             # C-alias: 0, 1, 2; 255 = outside the radar crop
                if tuple(a.shape) == tuple(shape):
                    return a, f"{what} ({hits[0]})", grades
                log.warning("%s is %s, the product %s: not used", hits[0], a.shape, shape)
            except Exception as exc:                      # noqa: BLE001
                log.warning("Could not read %s (%s)", hits[0], exc)
    return None, "no flag layer (F2 not written yet, no quality_too_fast_to_measure.tif)", False


# =============================================================== C-alias: the F2 flag per comparison interval

def _aliasing():
    return importlib.import_module(f"{__package__}.aliasing" if __package__ else "aliasing")


def read_alias_summary(cfg) -> dict | None:
    """deliverables/aliasing_risk.json (C-alias): the method, thresholds,
    lower-bound statement and shares, as written beside the flag layer. None
    when the file does not exist yet."""
    p = Path(cfg.project_dir) / "deliverables" / ALIAS_JSON
    if not p.exists():
        return None
    try:
        js = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return {"file": str(p), "error": f"not read ({exc})"}
    keep = ("version", "written", "lower_bound", "values", "thresholds", "method", "wavelength_m",
            "wavelength_source", "quarter_wavelength_mm", "look_cell_m", "tilt_limit_mm_per_m_per_interval", "route",
            "zones_geojson", "fast_zones", "product_grid", "overall_radar_crop", "old_flag", "harness",
            "record_check", "fast_zones_stale")
    out = {"file": str(p), "sha256": _sha256(p), **{k: js[k] for k in keep if k in js}}
    out["n_pairs"] = len(js.get("pairs") or [])
    return out


class IntervalFlags:
    """Protocol A's flag for one comparison interval (C-alias, F2): a point
    is flagged for the interval (d0, d1) when at least half of the reliable
    cells it was sampled from carry level 1 or 2 (the json's
    harness.flagged_values) in the highest level over the exported pairs
    spanning part of the interval (pair dates A < d1 and B > d0; protocol A
    comparisons.unflagged_failure_rate.flag_definition, and exactly
    aliasing.risk_between). The whole-record quality_aliasing_risk.tif is
    the same over every pair, so it flags more than any one interval does.

    The per-pair record (<project>/aliasing/pairs/, kept by cleanup level 2)
    is read once, into the highest level over the pairs spanning each
    interval between consecutive pair dates: a pair spans part of (d0, d1)
    exactly when it spans one of those intervals that overlaps (d0, d1).
    Nothing is raised: `ok` False and `why` say why there is no interval
    flag (no record yet, the diagnostic off, an API it does not find)."""

    def __init__(self, cfg, prod, flagged=ALIAS_FLAGGED, version: str = "amended"):
        self.ok, self.why, self.what, self.pairs, self.stale_files = False, None, None, 0, 0
        self.flagged = tuple(int(v) for v in (flagged or ALIAS_FLAGGED))
        # which level of the per-pair record: the coherence hole as amended (2026-10-08/p23-2) or as
        # first fixed (the record's level_v1; a record from before the amendment holds only that rule)
        self.version, self.hole_rule = version, None
        try:
            self._load(cfg, prod)
        except Exception as exc:                          # noqa: BLE001 - stub-tolerant: recorded, never raised
            self.ok, self.why = False, f"the per-pair aliasing record was not read ({type(exc).__name__}: {exc})"

    def _load(self, cfg, prod) -> None:
        al = _aliasing()
        missing = [n for n in ("record_dir", "load_levels", "geo_index") if not hasattr(al, n)]
        if missing:
            self.why = f"aliasing.py has no {', '.join(missing)}"
            return
        on = getattr(al, "enabled", None)
        if on is not None and not on(cfg):
            self.why = "aliasing_diagnostic is off for this config"
            return
        d = Path(al.record_dir(cfg)) / "pairs"
        names = sorted(f.stem for f in d.glob("*.npz") if re.fullmatch(r"\d{8}_\d{8}", f.stem)) if d.exists() else []
        # the pairs of the last pre-unwrap call (summary.json), as aliasing.risk_between and the whole-record
        # tif use: a pair file left in pairs/ by an earlier network or a partial measurement is not one of them
        self.stale_files = 0
        listed = getattr(al, "_summary_pairs", lambda _c: None)(cfg)
        if listed is not None:
            keep = set(listed)
            self.stale_files = sum(1 for p in names if p not in keep)
            names = [p for p in names if p in keep]
        if not names:
            self.why = f"no per-pair aliasing record ({d})"
            return
        dates = sorted({x for p in names for x in p.split("_")})
        pos = {x: k for k, x in enumerate(dates)}
        U = None
        hr = getattr(al, "record_hole_rule", None)
        if hr is not None:
            with np.load(d / f"{names[0]}.npz") as z:
                self.hole_rule = hr(z) or {"kind": "as first fixed (the record predates amendment 2026-10-08/p23-2)"}
        for p in names:
            try:
                got, L, _ = al.load_levels(cfg, [p], self.version)
            except TypeError:                             # an aliasing module without versions: its one level
                got, L, _ = al.load_levels(cfg, [p])
            if L is None or not got:
                continue
            lv = np.asarray(L[0]).ravel()
            if U is None:
                U = np.zeros((max(len(dates) - 1, 1), lv.size), np.uint8)
            if lv.size != U.shape[1]:
                raise ValueError(f"pair {p} is on another radar grid ({lv.size} cells, not {U.shape[1]})")
            a, b = pos[p[:8]], pos[p[9:]]
            if b > a:
                np.maximum(U[a:b], lv[None, :].astype(np.uint8), out=U[a:b])
        if U is None:
            self.why = f"the per-pair aliasing record ({d}) holds no level"
            return
        index = np.asarray(al.geo_index(cfg, prod.attrs, prod.shape))
        if tuple(index.shape) != tuple(prod.shape):
            raise ValueError(f"the radar-to-product index is {index.shape}, the product {prod.shape}")
        self.U, self.index, self.days = U, index, _days(dates)
        self.pairs, self.ok = len(names), True
        self.what = (f"F2 aliasing flag per comparison interval: the highest level over the {len(names)} exported "
                     f"pairs spanning part of the interval ({d}; aliasing.risk_between's rule; coherence hole "
                     + ("as amended, 2026-10-08/p23-2" if self.version == "amended" else "as first fixed") + ")"
                     + (f"; {self.stale_files} pair file(s) not in the record's summary.json left out"
                        if self.stale_files else ""))

    def levels(self, rows, cols, d0, d1) -> np.ndarray:
        """The interval's level at product cells (rows, cols); NaN outside the radar crop."""
        rows, cols = np.asarray(rows, int), np.asarray(cols, int)
        idx = self.index[rows, cols] if rows.size else np.array([], np.int64)
        out = np.full(idx.shape, np.nan)
        hit = idx >= 0
        if not hit.any():
            return out
        a, b = sorted((int(d0), int(d1)))
        ks = np.flatnonzero((self.days[:-1] < b) & (self.days[1:] > a))
        out[hit] = self.U[ks[:, None], idx[hit][None, :]].max(0) if ks.size else 0.0
        return out

    def point(self, rows, cols, d0, d1) -> dict:
        """The point's flag for the interval: flagged when >= half of its cells carry a flagged level."""
        v = self.levels(rows, cols, d0, d1)
        v = v[np.isfinite(v)]
        rec = {"interval": [_iso(min(d0, d1)), _iso(max(d0, d1))], "cells": int(v.size), "basis": "interval"}
        if not v.size:
            return {**rec, "flagged": None, "why": "no cell of the point inside the radar crop of the record"}
        sh = float(np.isin(v, self.flagged).mean())
        return {**rec, "flagged": bool(sh >= 0.5), "share": round(sh, 4), "likely_share": round(float((v >= 2).mean()), 4)}


# =============================================================== C-events: protocol A's peak-ratio rows

# a Tahmoor West panel W1..W4 in a feature's name or properties (not the 'W' of an Appin 'LW')
TW_PANEL_RE = re.compile(r"(?<![A-Za-z])W\s*-?\s*0*([1-4])(?!\d)", re.I)
_FEATURE_NAME_KEYS = ("name", "panel", "Name", "NAME", "id", "ID", "label", "line", "title")


def _events():
    return importlib.import_module(f"{__package__}.events" if __package__ else "events")


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _feature_text(ft) -> str:
    """Every plain property value of a GeoJSON feature, as one text."""
    p = (ft or {}).get("properties") or {}
    return " ".join(str(v) for v in p.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool))


def _names(text, token) -> bool:
    """Whether `token` (an id such as LW902, or ARTC) appears in `text`,
    ignoring case, spaces and punctuation, and not followed by another digit."""
    t, k = _norm(text), _norm(token)
    return bool(k) and re.search(re.escape(k) + r"(?!\d)", t) is not None


def _line_matches(ft, panel, key, value) -> bool:
    """Whether a survey-line feature is the one an end-of-panel maximum was
    measured on. A line that lists its maxima (properties
    end_of_panel_maxima: [{panel, max_mm, ...}], as the T1 digitising writes
    them) matches by panel and value; any other line by its properties naming
    the panel ('LW902') and, where the panel has several maxima, the line's
    first word ('ARTC', 'M31')."""
    eop = ((ft or {}).get("properties") or {}).get("end_of_panel_maxima")
    if isinstance(eop, list) and eop:
        for m in eop:
            if isinstance(m, dict) and _norm(m.get("panel", "")) == _norm(panel):
                mv = m.get("max_mm")
                if not _isnum(mv) or value is None or abs(float(mv) - float(value)) < 0.5:
                    return True
        return False
    text = _feature_text(ft)
    word = str(key).split("_")[0] if key else None
    return _names(text, panel) and (word is None or _names(text, word))


def _read_features(path, kinds) -> tuple[list, dict | None]:
    """The features of a GeoJSON whose geometry type is in `kinds`, and its legacy 'crs' member."""
    data = json.loads(Path(path).read_text())
    if data.get("type") == "FeatureCollection":
        feats = data.get("features") or []
    elif data.get("type") == "Feature":
        feats = [data]
    else:
        feats = []
    return ([f for f in feats if isinstance(f, dict) and ((f.get("geometry") or {}).get("type") in kinds)],
            data.get("crs"))


def event_inputs(P, prm, cfg, prod, outlines=None, lines=None, run=None) -> dict:
    """C-events, read once for protocol A's peak-ratio rows (the reference
    station's double difference and the grades come later, in _peak_rows):

      appin_peak_ratio      the LOS change along each end-of-panel survey line
                            (truth.end_of_panel_maxima.panels_in_window, the
                            panels this run covers) between its panel's
                            mining dates, snapped outward to the run's dates,
                            in the protocol's bins (events.profile: reliable
                            cells within 100 m of the line, 50 m bins, the
                            sampling minimum per bin). A line is found by its
                            properties naming the panel (e.g. 'LW902'); where
                            a panel has several maxima, also the line's first
                            word ('ARTC', 'M31').
      tahmoor_w_peak_ratio  the deepest robust subsidence over the run's
                            window inside each Tahmoor West panel outline W1-W4
                            (events.panel_max over the outlines named W1..W4).

    The survey lines and outlines are the digitised T1 files (--lines,
    --outlines), else the config's event_lines / event_polygons. Nothing is
    raised: a missing file, module or function is recorded in 'why', and the
    row then reads 'not computed' with that reason."""
    C = P.get("comparisons") or {}
    T = P.get("truth") or {}
    src_l = str(lines) if lines else (str(getattr(cfg, "event_lines", "") or "").strip() or None)
    src_o = str(outlines) if outlines else (str(getattr(cfg, "event_polygons", "") or "").strip() or None)
    out = {"lines_file": ({"path": src_l, "from": "--lines" if lines else "the config's event_lines"} if src_l else None),
           "outlines_file": ({"path": src_o, "from": "--outlines" if outlines else "the config's event_polygons"}
                             if src_o else None),
           "appin": [], "tahmoor": None, "why": {}}
    try:
        ev = _events()
    except Exception as exc:                              # noqa: BLE001 - stub-tolerant
        out["why"]["module"] = f"events.py could not be imported ({type(exc).__name__}: {exc})"
        return out
    floor = prm["floor"]
    lt = str((P.get("sampling") or {}).get("lines", ""))
    buf_m = _num(r"within\s+(\d+)\s*m", lt, 100.0)
    bin_m = _num(r"(\d+)\s*m\s+bins", lt, 50.0)
    first, last = int(prod.days[0]), int(prod.days[-1])

    spec = C.get("appin_peak_ratio")
    if isinstance(spec, dict) and _not_this_run(spec, run) is None:
        panels = (T.get("end_of_panel_maxima") or {}).get("panels_in_window") or {}
        # the protocol's named exception: "for A3/D1 at LW903 the start date is A3's first date"
        allowed = {(pn, r) for rs, pn in re.findall(r"for\s+([A-Za-z0-9/]+)\s+at\s+(LW\d+)", str(spec.get("insar_peak", "")))
                   for r in rs.split("/")}
        if not src_l:
            out["why"]["appin_peak_ratio"] = ("needs the digitised end-of-panel survey lines (PLAN T1): pass them with "
                                              "--lines, or set event_lines in the config")
        elif not Path(src_l).exists():
            out["why"]["appin_peak_ratio"] = f"the survey-line file {src_l} does not exist"
        elif not hasattr(ev, "profile"):
            out["why"]["appin_peak_ratio"] = "events.profile is not available"
        else:
            feats, crs = _read_features(src_l, ("LineString", "MultiLineString"))
            for pnl, rec in panels.items():
                if run and isinstance(rec, dict) and isinstance(rec.get("runs"), list) and run not in [str(x) for x in rec["runs"]]:
                    continue
                mx, mined = rec.get("max_mm"), list(rec.get("mined") or [])
                for key, val in (list(mx.items()) if isinstance(mx, dict) else [(None, mx)]):
                    e = {"panel": str(pnl), "line_key": key, "truth_max_mm": float(val), "mined": mined}
                    out["appin"].append(e)
                    word = str(key).split("_")[0] if key else None
                    fs = [f for f in feats if _line_matches(f, pnl, key, val)]
                    if not fs:
                        e["why"] = (f"no survey line in {Path(src_l).name} names {pnl}" + (f" and {word}" if word else "")
                                    + f" (or lists its {float(val):g} mm maximum)")
                        continue
                    e["line_ids"] = [str((f.get("properties") or {}).get("id") or (f.get("properties") or {}).get("name"))
                                     for f in fs]
                    if len(mined) != 2:
                        e["why"] = "the protocol gives no mining dates for the panel"
                        continue
                    d0, d1 = _day(mined[0]), _day(mined[1])
                    if d1 > last:
                        e["why"] = f"the panel ends {mined[1]}, after the run's last date {_iso(last)}"
                        continue
                    if d0 < first:
                        if (str(pnl), str(run)) not in allowed:
                            e["why"] = f"the panel starts {mined[0]}, before the run's first date {_iso(first)}"
                            continue
                        e["start_note"] = (f"the panel started {mined[0]}, before the run's first date {_iso(first)}: "
                                           f"measured from that date, {first - d0} days into the panel (the protocol's "
                                           "stated exception)")
                        d0 = first
                    fc = {"type": "FeatureCollection", "features": fs, **({"crs": crs} if crs else {})}
                    try:
                        df = ev.profile(cfg, fc, [_iso(d0), _iso(d1)], buffer_m=buf_m, bin_m=bin_m, floor=floor,
                                        snap="outer", min_cells=prm["min_cells"],
                                        min_independent=prm["min_independent_pixels"])
                    except Exception as exc:              # noqa: BLE001 - recorded, the row says why
                        e["why"] = f"events.profile failed ({type(exc).__name__}: {exc})"
                        continue
                    e["lines"] = sorted({str(x) for x in df["line"]}) if len(df) else []
                    e["bins"] = [{k: r.get(k) for k in ("line", "bin", "distance_m", "lat", "lon", "baseline_epoch",
                                                        "epoch", "change_mm", "vertical_assumed_mm", "n_cells",
                                                        "n_independent", "status")}
                                 for r in df.to_dict("records")]
                    e["settings"] = {k: v for k, v in dict(getattr(df, "attrs", {}) or {}).items()
                                     if isinstance(v, (str, int, float))}

    spec = C.get("tahmoor_w_peak_ratio")
    if isinstance(spec, dict) and _not_this_run(spec, run) is None:
        tw = {"truth_mm": _f((T.get("tahmoor_west_peak") or {}).get("value_mm"), 1)}
        out["tahmoor"] = tw
        if not src_o:
            tw["why"] = ("needs the digitised panel outlines (PLAN T1): pass them with --outlines, or set "
                         "event_polygons in the config")
        elif not Path(src_o).exists():
            tw["why"] = f"the outline file {src_o} does not exist"
        elif not hasattr(ev, "panel_max"):
            tw["why"] = "events.panel_max is not available"
        else:
            feats, crs = _read_features(src_o, ("Polygon", "MultiPolygon"))
            sel = []
            for f in feats:
                m = TW_PANEL_RE.search(_feature_text(f))
                if m:
                    p = f.get("properties") or {}
                    nm = next((str(p[k]) for k in _FEATURE_NAME_KEYS if p.get(k) not in (None, "")), f"W{m.group(1)}")
                    # only the name: a panel's own dates would replace the run's window in panel_max
                    sel.append({"type": "Feature", "geometry": f["geometry"], "properties": {"name": nm}})
            if not sel:
                tw["why"] = f"no outline in {Path(src_o).name} is named W1..W4 (Tahmoor West)"
            else:
                w0, w1 = prm["window"]
                try:
                    tw["rows"] = ev.panel_max(cfg, {"type": "FeatureCollection", "features": sel,
                                                    **({"crs": crs} if crs else {})},
                                              windows=[{"name": "run window", "start": w0, "end": w1}], floor=floor,
                                              min_cells=prm["min_cells"])
                except Exception as exc:                  # noqa: BLE001 - recorded, the row says why
                    tw["why"] = f"events.panel_max failed ({type(exc).__name__}: {exc})"
    return out


def _peak_rows(C, prm, prod, samples, R, ev_in, alias, flag) -> dict:
    """protocol A's appin_peak_ratio (graded) and tahmoor_w_peak_ratio
    (report only) from event_inputs, vertical (LOS / cos(incidence),
    labelled: the D1 Up is not read along lines) and double-differenced to
    the reference station R: R's own vertical change over the same SAR
    dates is taken off. The peak of a line is the 5th percentile of its
    binned values (a robust minimum); ratio = -peak / line maximum. Every
    panel-line must pass for the row to pass (the row reads the lowest
    ratio's grade); line maxima are lower bounds, so a ratio above 1 is
    reported, not a failure. Each line maximum also joins the
    honest-or-flagged row ('_comparisons'): its error is the InSAR peak's
    shortfall against the line maximum (0 when the InSAR is deeper), masked
    when no bin has a value, flagged by the cells within the sampling radius
    of the peak's bin over the panel's dates."""
    out = {"_comparisons": []}
    fl = prm["floor"]
    pos = {int(d): i for i, d in enumerate(prod.days)}
    IR = samples[R][fl]["series"] if R is not None and R in samples else None
    luR = float(samples[R][fl]["los"][2]) if IR is not None else float("nan")
    layer = flag[0] if flag else None

    def ref_up(e0, e1):
        try:
            i0, i1 = pos.get(_day(e0)), pos.get(_day(e1))
        except (TypeError, ValueError):
            return float("nan")
        if IR is None or i0 is None or i1 is None or not math.isfinite(luR) or luR <= 0.1:
            return float("nan")
        return float((IR[i1] - IR[i0]) / luR)

    def flag_at(lat, lon, e0, e1):
        r, c, _ = prod.cells_near(float(lat), float(lon), prm["radius_m"])
        rel = np.isfinite(prod.tcoh[r, c]) & (prod.tcoh[r, c] >= fl) if r.size else np.array([], bool)
        r, c = r[rel], c[rel]
        if alias is not None and alias.ok:
            return alias.point(r, c, _day(e0), _day(e1))
        if layer is not None and r.size:
            v = layer[r, c]
            v = v[np.isfinite(v)]
            return {"flagged": bool(v.size and np.mean(v > 0) >= 0.5), "cells": int(v.size), "basis": "whole record"}
        return {"flagged": None, "cells": int(r.size), "basis": "no flag layer"}

    spec = C.get("appin_peak_ratio")
    if isinstance(spec, dict) and (ev_in.get("appin") or ev_in.get("why", {}).get("appin_peak_ratio")):
        ents = []
        for e in ev_in.get("appin") or []:
            r = {k: e[k] for k in ("panel", "line_key", "truth_max_mm", "mined", "start_note", "why", "lines", "line_ids")
                 if k in e}
            ents.append(r)
            if e.get("why"):
                continue
            truth = -abs(float(e["truth_max_mm"]))
            name = f"{e['panel']}" + (f" {e['line_key']}" if e.get("line_key") else " line")
            ok = [b for b in e.get("bins") or [] if b.get("status") == "ok" and b.get("vertical_assumed_mm") is not None]
            r.update(bins_total=len(e.get("bins") or []), bins_with_value=len(ok))
            if not ok:
                ep = [b for b in e.get("bins") or []]
                on_grid = any(prod.cells_near(float(b["lat"]), float(b["lon"]), prm["radius_m"])[0].size
                              for b in ep if b.get("lat") is not None and b.get("lon") is not None)
                if not on_grid:
                    r["why"] = "the line lies outside the product grid: no comparison"
                    continue
                r["why"] = "no bin of the line has a reliable value at both dates (masked: honest, not a ratio)"
                out["_comparisons"].append({"row": "appin_peak_ratio", "point": name, "kind": "line_max",
                                            "since": ep[0].get("baseline_epoch"), "date": ep[0].get("epoch"),
                                            "insar": None, "truth": _f(truth, 1), "error": None, "masked": True,
                                            "lower_bound_truth": True})
                continue
            e0, e1 = ok[0]["baseline_epoch"], ok[0]["epoch"]
            rv = ref_up(e0, e1)
            if not math.isfinite(rv):
                r["why"] = (f"the reference station {R} has no InSAR value at {e0} or {e1}" if R is not None else
                            "no reference station is usable (detail.reference_choice), so there is no datum")
                continue
            v = np.array([float(b["vertical_assumed_mm"]) for b in ok]) - rv
            peak = float(np.percentile(v, 5))
            b = ok[int(np.argmin(np.abs(v - peak)))]
            ratio = -peak / abs(truth)
            M = {"ratio": ratio, "peak_mm": peak, "truth_max_mm": abs(truth)}
            g = grade_row(spec.get("grade"), M)
            r.update(epochs=[e0, e1], reference=R, reference_up_change_mm=_f(rv, 1), insar_peak_up_mm=_f(peak, 1),
                     ratio=_f(ratio, 3), grade=g["grade"], label=g.get("label"),
                     peak_at={"line": b.get("line"), "distance_m": b.get("distance_m"), "lat": b.get("lat"),
                              "lon": b.get("lon")},
                     product_datum_peak_mm=_f(float(np.percentile(v + rv, 5)), 1),
                     label_vertical="LOS / cos(incidence): assumes vertical motion (D1 Up is not read along lines)")
            fg = flag_at(b["lat"], b["lon"], e0, e1)
            out["_comparisons"].append({"row": "appin_peak_ratio", "point": name, "kind": "line_max", "since": e0,
                                        "date": e1, "insar": _f(peak, 1), "truth": _f(truth, 1),
                                        "error": _f(max(0.0, peak - truth), 1), "masked": False,
                                        "flagged": fg.get("flagged"), "flag": fg, "lower_bound_truth": True})
        graded = [x for x in ents if x.get("grade") in ("pass", "marginal", "fail")]
        rank = {"fail": 0, "marginal": 1, "pass": 2}
        if graded:
            worst = min(graded, key=lambda x: (rank[x["grade"]], x["ratio"]))
            M = {"ratio": min(x["ratio"] for x in graded), "n": len(graded), "n_expected": len(ents)}
            row = {"what": spec.get("what"), **_labelled(worst["grade"]), "metrics": M, "panels": ents,
                   "rule": "every panel-line must pass: the row reads the lowest ratio's grade; line maxima are lower "
                           "bounds, so a ratio above 1 is reported, not a failure"}
            missing = [x for x in ents if x not in graded]
            if missing:
                row["not_graded"] = [f"{x['panel']}{' ' + x['line_key'] if x.get('line_key') else ''}: {x.get('why')}"
                                     for x in missing]
        else:
            why = ev_in.get("why", {}).get("appin_peak_ratio") or "; ".join(
                f"{x['panel']}{' ' + x['line_key'] if x.get('line_key') else ''}: {x.get('why')}" for x in ents) \
                or "no end-of-panel maximum applies to this run"
            row = {"what": spec.get("what"), **_labelled("not_computed" if ev_in.get("why", {}).get("appin_peak_ratio")
                                                         else "not_testable", why), "panels": ents}
        row["lines_file"] = ev_in.get("lines_file")
        out["appin_peak_ratio"] = row

    spec = C.get("tahmoor_w_peak_ratio")
    tw = ev_in.get("tahmoor")
    if isinstance(spec, dict) and tw is not None:
        if tw.get("why"):
            out["tahmoor_w_peak_ratio"] = {"what": spec.get("what"), **_labelled("not_computed", tw["why"]),
                                           "outlines_file": ev_in.get("outlines_file")}
        else:
            per = []
            for p in tw.get("rows") or []:
                rec = {"panel": p.get("panel"), "max_mm_los": p.get("max_mm"), "date_of_max": p.get("date_of_max"),
                       "window": p.get("window"), "lat": p.get("lat"), "lon": p.get("lon"),
                       "max_mm_relative_to_ring": p.get("max_mm_relative_to_ring")}
                v = p.get("vertical_assumed_mm")
                e0 = (p.get("window") or {}).get("epoch_start")
                if p.get("max_mm") is None or v is None or e0 is None or p.get("date_of_max") is None:
                    rec["why"] = p.get("reason") or "no vertical value"
                    per.append(rec)
                    continue
                rv = ref_up(e0, p["date_of_max"])
                rec["reference_up_change_mm"] = _f(rv, 1)
                rec["insar_peak_up_mm"] = _f(float(v) - rv, 1) if math.isfinite(rv) else None
                if not math.isfinite(rv):
                    rec["why"] = (f"the reference station {R} has no InSAR value at {e0} or {p['date_of_max']}"
                                  if R is not None else "no reference station is usable, so there is no datum")
                per.append(rec)
            vals = [x["insar_peak_up_mm"] for x in per if x.get("insar_peak_up_mm") is not None]
            truth = tw.get("truth_mm")
            if vals and truth:
                peak = min(vals)
                M = {"ratio": -peak / abs(float(truth)), "peak_mm": peak, "truth_mm": abs(float(truth))}
                out["tahmoor_w_peak_ratio"] = {"what": spec.get("what"), **_labelled("report_only"), "metrics": M,
                                               "panels": per, "outlines_file": ev_in.get("outlines_file"),
                                               "label_vertical": "LOS / cos(incidence): assumes vertical motion",
                                               "note": "the deepest panel's robust peak (5th percentile) over the "
                                                       "run's window, double-differenced to the reference station; "
                                                       "C-band is expected to underestimate it (protocol A)"}
            else:
                whys = "; ".join(sorted({f"{x.get('panel')}: {x['why']}" for x in per if x.get("why")}))
                out["tahmoor_w_peak_ratio"] = {"what": spec.get("what"),
                                               **_labelled("not_testable", ("no W1-W4 outline has a peak value"
                                                                            + (f" ({whys})" if whys else ""))
                                                           if truth else "the protocol gives no Tahmoor West peak"),
                                               "panels": per, "outlines_file": ev_in.get("outlines_file")}
    return out


def events_crosscheck(cfg, prm, samples, comps, prod, coords) -> dict:
    """C-events' values_at at the truth points and epochs the harness
    compared, against the harness's own sampling (the same rule: median of
    the reliable cells within the radius, the sampling minimum, linear
    between SAR dates at most survey_max_gap_days apart). The two should
    agree to rounding; a disagreement is a warning, never a grade."""
    want: dict = {}
    for c in comps:
        if c.get("kind") not in ("event", "point_epoch") or c.get("masked") or c.get("error") is None:
            continue
        p = c["point"]
        if p not in samples:
            continue
        a, b = (c.get("t0"), c.get("t1")) if c.get("kind") == "event" else (c.get("since"), c.get("date"))
        if a and b:
            want.setdefault(p, set()).update({_iso(_day(a)), _iso(_day(b))})
    if not want:
        return {"n": 0, "why": "no event or digitised comparison with a value"}
    try:
        ev = _events()
        pts = {p: {"lat": float(coords[p]["lat"]), "lon": float(coords[p]["lon"])} for p in want if p in coords}
        if not pts:
            return {"n": 0, "why": "no coordinates for the compared points"}
        df = ev.values_at(cfg, pts, {p: sorted(want[p]) for p in pts}, prm["radius_m"], floor=prm["floor"],
                          min_cells=prm["min_cells"], min_independent=prm["min_independent_pixels"],
                          max_gap_days=prm["survey_max_gap_days"])
    except Exception as exc:                              # noqa: BLE001 - stub-tolerant
        return {"n": 0, "why": f"events.values_at not used ({type(exc).__name__}: {exc})"}
    got = {(str(r["id"]), str(r["epoch"])): r.get("value_mm") for r in df.to_dict("records")}
    sar = prod.days.astype(float)
    diffs = []
    for p, eps in want.items():
        if p not in pts:
            continue
        s = samples[p][prm["floor"]]["series"]
        for e in eps:
            h = _interp_cols(np.asarray(s, float)[:, None], sar, float(_day(e)), prm["survey_max_gap_days"])[0]
            v = got.get((p, e))
            if v is None or not math.isfinite(h):
                continue
            diffs.append(abs(float(v) - float(h)))
    if not diffs:
        return {"n": 0, "why": "no common value to compare"}
    mx = max(diffs)
    return {"n": len(diffs), "max_abs_diff_mm": round(mx, 3), "agree": bool(mx <= 0.05),
            "what": "events.values_at against the harness's own sampling at the truth points' compared epochs (LOS, "
                    "product datum, interpolated alike); they share the rule, so they should agree to rounding"}


def run_reference(P, prm, prod, sources, samples, floors, points_truth=None, flag=(None, "", False), up=None,
                  stable=None, alias=None, events_in=None) -> dict:
    """Reference-station design (protocol A): CA19-type event changes and
    matched series double-differenced to the reference station, null-test
    stations, digitised points, and the honest-or-flagged row. With
    `stable` (stable_placements; amendment 2026-10-08/1) every event change
    and digitised unit-epoch of the primary reference also gets its 95%
    interval from the stable ground. With `alias` (IntervalFlags, C-alias)
    every comparison is flagged for its own interval, as protocol A's flag
    definition says; without it the whole-record flag layer stands in. With
    `events_in` (event_inputs, C-events) the peak-ratio rows are made and
    each end-of-panel line maximum joins the honest-or-flagged row."""
    sar = prod.days
    half = prm["match_days"]
    C = P.get("comparisons") or {}
    sw = prm.get("amend") or {}
    if stable is not None and stable.get("series") is None:
        stable = None
    # the stable ground serves two amendments: the event changes' 95% interval (2026-10-08/1) and the
    # flag's selectivity (2026-10-08/p23-2); each only when its own key is set
    stable_iv = stable if sw.get("event_change_sigma") == "stable_ground" else None
    flag_layer, flag_what, flag_grades = flag
    match = {src: {s: match_epochs(st, sar, half) for s, st in sources.get(src, {}).items()} for src in sources}
    result = {"design": "reference_station", "by_floor": {}, "flag_layer": flag_what, "epoch_table": []}

    def src_for(p, r):
        for src in prm["source_order"]:
            if p in match.get(src, {}) and r in match.get(src, {}):
                return src
        return None

    event_rows = {n: c for n, c in C.items() if isinstance(c, dict) and _event_dates(c.get("epochs"))}
    series_rows = {n: c for n, c in C.items() if isinstance(c, dict) and c.get("statistic") == "rmse_after_offset"}

    for fl in floors:
        primary = fl == prm["floor"]
        I = {s: samples[s][fl]["series"] for s in samples if not s.startswith("_")}
        L = {s: samples[s][fl]["los"] for s in samples if not s.startswith("_")}
        entry = {"rows": {}, "reference_choice": {}, "comparisons": []}
        # ---- choose the reference on coverage alone, before any difference. The compared epochs
        # are those of the matched-series rows (CA19) inside this run's dates; a run that has none
        # (A3 and D1 start 2019-11, after CA19's GA record) is judged on its own dates in the window,
        # so the Tahmoor West units still get the protocol's reference instead of none at all.
        compared, skipped = {}, []
        for name, c in series_rows.items():
            span = _series_span(c.get("epochs"))
            pts = [s for s in _stations_named(c.get("what"), prm["stations"]) if s in prm.get("validation_stations", [])]
            if span and pts:
                a, b = _day(span[0]), _day(span[1])
                why_skip = _not_this_run(c, prm.get("run"))
                if why_skip is None and not ((sar >= a) & (sar <= b)).any():
                    why_skip = f"none of this run's dates lies in {span[0]} .. {span[1]}"
                if why_skip:
                    skipped.append({"row": name, "why": why_skip})
                    continue
                compared[name] = (pts[0], a, b)
        wa, wb = (_day(x) if x else None for x in prm["window"])
        in_win = (sar >= (wa if wa is not None else sar.min())) & (sar <= (wb if wb is not None else sar.max()))
        cands = [prm["reference_station"]] + prm.get("reference_fallback", [])
        choice, table = None, []
        for R in cands:
            ok_all, rows = True, []
            if not compared:
                src = next((s for s in prm["source_order"] if R in match.get(s, {})), None)
                n = int(in_win.sum())
                if src is None or R not in I or not n:
                    ok_all = False
                    rows.append({"row": "this run's dates in the window",
                                 "why": "no truth for the station, no InSAR sample, or no date in the window"})
                else:
                    fi = float(np.isfinite(I[R][in_win]).mean())
                    fg = float(np.all(np.isfinite(match[src][R][0][in_win]), 1).mean())
                    ok_all = fi >= prm["usable_share"] and fg >= prm["usable_share"]
                    rows.append({"row": "this run's dates in the window", "source": src, "compared_epochs": n,
                                 "insar_share": _f(fi, 3), "gnss_share": _f(fg, 3), "usable": ok_all})
            for name, (p, a, b) in compared.items():
                src = src_for(p, R)
                if src is None or R not in I:
                    ok_all = False
                    rows.append({"row": name, "why": "no truth from one source for both stations, or no InSAR sample"})
                    continue
                Ep = match[src][p][0]
                ep = (sar >= a) & (sar <= b) & np.all(np.isfinite(Ep), 1)
                n = int(ep.sum())
                fi = float(np.isfinite(I[R][ep]).mean()) if n else 0.0
                fg = float(np.all(np.isfinite(match[src][R][0][ep]), 1).mean()) if n else 0.0
                ok = n > 0 and fi >= prm["usable_share"] and fg >= prm["usable_share"]
                ok_all &= ok
                rows.append({"row": name, "source": src, "compared_epochs": n, "insar_share": _f(fi, 3),
                             "gnss_share": _f(fg, 3), "usable": ok})
            table.append({"station": R, "usable": ok_all, "checks": rows})
            if ok_all and choice is None:
                choice = R
        entry["reference_choice"] = {"rule": f"first of {cands} with an InSAR value and GNSS at >= "
                                             f"{prm['usable_share']:.0%} of the compared epochs"
                                             + ("" if compared else " (no matched-series row lies in this run's "
                                                                    "dates: the compared epochs are the run's "
                                                                    "own dates in the window)"),
                                     "rows_not_in_this_run": skipped,
                                     "candidates": table, "chosen": choice}
        if not compared and sw.get("reference_epochs_without_series_row") == "run_dates_in_window":
            entry["reference_choice"]["amendment"] = "2026-10-08/6 (the run's own dates, confirmed)"
        if choice is None:
            parts = []
            for t_ in table:
                bad = [c_ for c_ in t_["checks"] if not c_.get("usable")]
                parts.append(f"{t_['station']}: " + ("; ".join(
                    c_.get("why") or f"InSAR at {100 * (c_.get('insar_share') or 0):.0f}%, GNSS at "
                                     f"{100 * (c_.get('gnss_share') or 0):.0f}% of {c_.get('compared_epochs')} compared epochs"
                    for c_ in bad) if bad else "no compared epochs"))
            why_ref = (f"no reference station is usable (needs an InSAR value and GNSS at >= "
                       f"{prm['usable_share']:.0%} of the compared epochs): " + " | ".join(parts))
            for name in list(event_rows) + list(series_rows):
                entry["rows"][name] = {"what": C[name].get("what"), **_labelled("not_testable", why_ref)}
        refs = [choice] + ([s for s in prm.get("reference_sensitivity", []) if s != choice] if primary else [])
        for R in [r for r in refs if r]:
            sens = R != choice
            key = "primary" if not sens else f"reference_{R}"
            comps, rows = [], {}
            # ---- event changes
            for name, c in event_rows.items():
                d0, d1 = (_day(x) for x in _event_dates(c.get("epochs")))
                pts = [s for s in _stations_named(c.get("what"), prm["stations"]) if s != R]
                if not pts:
                    continue
                p = pts[0]
                src = src_for(p, R)
                rec = {"row": name, "point": p, "reference": R, "source": src, "kind": "event"}
                if src is None or p not in I:
                    rows[name] = {"what": c.get("what"), **_labelled("not_testable", f"no truth for {p} and {R} from one source"
                                                                     if src is None else f"{p} has no InSAR sample")}
                    continue
                Ep, Er = match[src][p][0], match[src][R][0]
                okp = np.all(np.isfinite(Ep), 1)
                i0 = np.flatnonzero((sar <= d0) & okp)
                i1 = np.flatnonzero((sar >= d1) & okp)
                if not i0.size or not i1.size:
                    rows[name] = {"what": c.get("what"), **_labelled("not_testable", f"no SAR date with {p} truth within "
                                                                     f"+-{half} d on the needed side of {_iso(d0)} / {_iso(d1)}")}
                    continue
                t0, t1 = int(i0[-1]), int(i1[0])
                rec.update(t0=prod.dates[t0], t1=prod.dates[t1])
                if not (np.all(np.isfinite(Er[t0])) and np.all(np.isfinite(Er[t1]))):
                    rows[name] = {"what": c.get("what"), **_labelled("not_testable", f"no {R} truth at {prod.dates[t0]} "
                                                                     f"or {prod.dates[t1]}"), "dates": [prod.dates[t0], prod.dates[t1]]}
                    continue
                dE = (Ep[t1] - Ep[t0]) - (Er[t1] - Er[t0])
                truth = float(L[p] @ dE)
                dI = (I[p][t1] - I[p][t0]) - (I[R][t1] - I[R][t0])
                # masked = the POINT's cells have no reliable value (honest); a gap at the reference
                # station is no comparison at all, and must not count as the point being honest
                masked = not (math.isfinite(I[p][t0]) and math.isfinite(I[p][t1]))
                if not masked and not math.isfinite(dI):
                    rec.update(t0=prod.dates[t0], t1=prod.dates[t1], truth=_f(truth, 1), insar=None, error=None,
                               masked=False, no_comparison=f"{R} has no InSAR value at {prod.dates[t0]} or {prod.dates[t1]}")
                    comps.append(rec)
                    rows[name] = {"what": c.get("what"), **_labelled("not_testable", rec["no_comparison"]),
                                  "comparison": rec}
                    continue
                sgp, _ = daily_sigma_and_range(sources[src][p], L[p], prm)
                sgr, _ = daily_sigma_and_range(sources[src][R], L[p], prm)
                s_truth = math.sqrt(2 * sgp ** 2 + 2 * sgr ** 2) if math.isfinite(sgp + sgr) else None
                err = dI - truth if not masked else None
                rec.update(insar=_f(dI, 1), truth=_f(truth, 1), error=_f(err, 1), masked=masked, truth_sigma=_f(s_truth, 1),
                           truth_up=_f(float(dE[2]), 1), truth_east=_f(float(dE[0]), 1), truth_north=_f(float(dE[1]), 1))
                lu_p, lu_r = L[p][2], L[R][2]
                dIv = (I[p][t1] - I[p][t0]) / lu_p - (I[R][t1] - I[R][t0]) / lu_r
                rec["vertical_assumed"] = {"insar_up": _f(dIv, 1), "truth_up": _f(float(dE[2]), 1),
                                           "error": _f(dIv - float(dE[2]), 1) if math.isfinite(dIv) else None,
                                           "label": "LOS / cos(incidence): assumes vertical motion"}
                if alias is not None and alias.ok and not masked:
                    rec["flag"] = alias.point(samples[p][fl]["rows"], samples[p][fl]["cols"], sar[t0], sar[t1])
                    rec["flagged"] = rec["flag"]["flagged"]
                if stable_iv is not None and primary and not sens:
                    # amendment 2026-10-08/1: the same double-differenced LOS change on stable ground
                    dQ = (stable_iv["series"][t1] - stable_iv["series"][t0]) - (I[R][t1] - I[R][t0])
                    rl = prm["stations"][R]
                    pl = prm["stations"][p]
                    _interval_fields(rec, stable_change_sigma(
                        dQ, _km_between(rl["lat"], rl["lon"], stable_iv["lat"], stable_iv["lon"]),
                        float(_km_between(rl["lat"], rl["lon"], pl["lat"], pl["lon"])), sw), s_truth)
                comps.append(rec)
                M = {"abs_err": abs(err) if err is not None else None, "truth": truth, "err": err}
                g = grade_row(c.get("grade"), M) if not masked else _labelled("not_testable", "no InSAR value at a date (masked)")
                rows[name] = {"what": c.get("what"), **g, "metrics": M, "comparison": rec}
            # ---- matched series
            for name, c in series_rows.items():
                span = _series_span(c.get("epochs"))
                pts = [s for s in _stations_named(c.get("what"), prm["stations"]) if s != R]
                if not span or not pts:
                    continue
                p = pts[0]
                src = src_for(p, R)
                if src is None or p not in I:
                    rows[name] = {"what": c.get("what"), **_labelled("not_testable", "no common truth source or no InSAR sample")}
                    continue
                Ep, Er = match[src][p][0], match[src][R][0]
                a, b = _day(span[0]), _day(span[1])
                ep = (sar >= a) & (sar <= b) & np.all(np.isfinite(Ep), 1) & np.all(np.isfinite(Er), 1)
                dI = I[p] - I[R]
                dG = (Ep - Er) @ L[p]
                m = ep & np.isfinite(dI)
                p_masked = ep & ~np.isfinite(I[p])                       # the point's own cells: honest
                r_gap = ep & np.isfinite(I[p]) & ~np.isfinite(I[R])       # the reference's: no comparison
                n_masked = int(p_masked.sum())
                if m.sum() < 3:
                    rows[name] = {"what": c.get("what"), **_labelled("not_testable", f"{int(m.sum())} matched epochs")}
                    continue
                diff = (dI - dG)[m]
                off = float(diff.mean())
                res = diff - off
                rmse = float(np.sqrt(np.mean(res ** 2)))
                r = pearson(dI[m], dG[m])
                ci = bootstrap_ci(int(m.sum()), lambda idx, _r: float(np.sqrt(np.mean((res[idx] - res[idx].mean()) ** 2))),
                                  prm["n_boot"], prm["seed"])
                # each matched epoch as a comparison for the honest-or-flagged row: truth = the change
                # since the first compared epoch, error = the residual after the one offset
                i_first, i_last = int(np.flatnonzero(ep)[0]), int(np.flatnonzero(ep)[-1])
                g0 = dG[i_first]
                # C-alias: a matched epoch's error is a residual after ONE offset fitted over every compared
                # epoch, so whatever aliased anywhere in the series reaches it: its interval is the series' span
                s_flag = (alias.point(samples[p][fl]["rows"], samples[p][fl]["cols"], sar[i_first], sar[i_last])
                          if alias is not None and alias.ok else None)
                for i in np.flatnonzero(m):
                    e_i = dI[i] - off - dG[i]
                    rec_i = {"row": name, "point": p, "reference": R, "kind": "matched_epoch", "date": prod.dates[i],
                             "since": prod.dates[i_first], "insar": _f(dG[i] - g0 + e_i, 1), "truth": _f(dG[i] - g0, 1),
                             "error": _f(e_i, 1), "masked": False}
                    if s_flag is not None:
                        rec_i["flag"] = {**s_flag, "interval_is": "the series' span (one offset over all its epochs)"}
                        rec_i["flagged"] = s_flag["flagged"]
                    comps.append(rec_i)
                for i in np.flatnonzero(p_masked):
                    comps.append({"row": name, "point": p, "reference": R, "kind": "matched_epoch", "date": prod.dates[i],
                                  "insar": None, "truth": _f(dG[i] - g0, 1), "error": None, "masked": True})
                for i in np.flatnonzero(r_gap):
                    comps.append({"row": name, "point": p, "reference": R, "kind": "matched_epoch", "date": prod.dates[i],
                                  "insar": None, "truth": _f(dG[i] - g0, 1), "error": None, "masked": False,
                                  "no_comparison": f"{R} has no InSAR value at this date"})
                M = {"rmse": rmse, "r": r, "n": int(m.sum()), "offset_removed_mm": off}
                rows[name] = {"what": c.get("what"), **grade_row(c.get("grade"), M), "metrics": M,
                              "ci": {"rmse": ci, "note": "epochs resampled within the one point (bootstrap over points "
                                                         "needs more than one)"},
                              "masked_epochs": n_masked, "reference_gap_epochs": int(r_gap.sum()), "source": src,
                              "_series": {"dates": [prod.dates[i] for i in np.flatnonzero(m)],
                                          "insar": (dI[m] - off).tolist(), "truth": dG[m].tolist()}}
                if primary and not sens:
                    for i in np.flatnonzero(ep):
                        result["epoch_table"].append({"station": p, "reference": R, "date": prod.dates[i],
                                                      "insar_dd_los_mm": _f(dI[i], 2), "gnss_dd_los_mm": _f(dG[i], 2),
                                                      "n_cells": int(samples[p][fl]["n_cells"][i]),
                                                      "n_independent": int(samples[p][fl]["n_independent"][i])})
            # ---- null-test stations (report only)
            nulls = {}
            for s in prm.get("null_stations", []):
                if s == R or s not in I:
                    continue
                src = src_for(s, R)
                if src is None:
                    nulls[s] = {"why": "no truth from one source for both stations"}
                    continue
                dI = I[s] - I[R]
                dG = (match[src][s][0] - match[src][R][0]) @ L[s]
                m = np.isfinite(dI) & np.isfinite(dG)
                rI = model_rates(sar[m], dI[m], dict(prm, water_years=[]))
                rG = model_rates(sar[m], dG[m], dict(prm, water_years=[]))
                li, lg = rI["rates"].get("linear"), rG["rates"].get("linear")
                cum = (float(dI[m][-1] - dI[m][0]), float(dG[m][-1] - dG[m][0])) if m.sum() >= 2 else (None, None)
                nulls[s] = {"epochs": int(m.sum()), "insar_rate": _f((li or {}).get("rate"), 2),
                            "insar_rate_sigma": _f((li or {}).get("sigma"), 2), "gnss_rate": _f((lg or {}).get("rate"), 2),
                            "gnss_rate_sigma": _f((lg or {}).get("sigma"), 2),
                            "insar_change": _f(cum[0], 1), "gnss_change": _f(cum[1], 1)}
            if prm.get("null_stations"):
                rows["stable_stations"] = {"what": (C.get("stable_stations") or {}).get("what"),
                                           **_labelled("report_only"), "stations": nulls}
            entry["comparisons" if not sens else key + "_comparisons"] = comps
            entry["rows" if not sens else key] = rows
        result["by_floor"][fl] = entry

    # ---- rows the harness cannot compute yet, or that need the flag / digitised truth
    prim = result["by_floor"][prm["floor"]]
    rows = prim["rows"]
    comps = prim["comparisons"]
    if "tahmoor_w_units" in C:
        if points_truth:
            rows["tahmoor_w_units"] = _points_row(C["tahmoor_w_units"], prm, prod, samples, points_truth,
                                                  sources, prim["reference_choice"].get("chosen"), flag, comps,
                                                  stable=stable_iv, alias=alias)
        else:
            rows["tahmoor_w_units"] = {"what": C["tahmoor_w_units"].get("what"),
                                       **_labelled("not_computed", "needs the digitised Tahmoor West units (PLAN T1, "
                                                                   "phase 2): pass them with --points")}
    R_ = prim["reference_choice"].get("chosen")
    peak_rows = _peak_rows(C, prm, prod, samples, R_, events_in, alias, flag) if events_in is not None else {}
    for c_ in peak_rows.pop("_comparisons", []):
        comps.append(c_)
    rows.update(peak_rows)
    for name, why in (("appin_peak_ratio", "needs the digitised end-of-panel survey lines (PLAN T1, phase 2): "
                                           "pass them with --lines"),
                      ("tahmoor_w_peak_ratio", "needs the digitised panel outlines (PLAN T1, phase 2): pass them "
                                               "with --outlines"),
                      ("stable_ring", "needs the digitised panel outlines and draw distances (PLAN T1, phase 2)")):
        if name in C and name not in rows:
            rows[name] = {"what": C[name].get("what"), **_labelled("not_computed", why)}
    if "decomposition" in C and up is None:
        rows["decomposition"] = {"what": C["decomposition"].get("what"),
                                 **_labelled("not_computed", "needs the D1 Up/East product: pass --up")}
    elif up is not None:
        rows["decomposition_up_at_stations"] = _d1_reference_rows(prm, up, sources, match_src=prm["source_order"],
                                                                  R=prim["reference_choice"].get("chosen"))
    if "vertical_only_bias" in C and up is None:
        rows["vertical_only_bias"] = {"what": C["vertical_only_bias"].get("what"),
                                      **_labelled("not_computed", "needs the D1 Up product: pass --up")}
    if "vertical_secondary" in C:
        rows["vertical_secondary"] = {"what": C["vertical_secondary"].get("what"), **_labelled("report_only"),
                                      "events": [{"row": c["row"], **c["vertical_assumed"]} for c in comps
                                                 if c.get("kind") == "event" and "vertical_assumed" in c]}
    if "unflagged_failure_rate" in C:
        rows["unflagged_failure_rate"] = _unflagged_row(C["unflagged_failure_rate"], comps, prod, samples,
                                                        prm["floor"], flag, alias, stable=stable, sw=sw,
                                                        run=prm.get("run"))
    return result


def _d1_reference_rows(prm, up, sources, match_src, R) -> dict:
    """D1 Up at each validation and null-test station, double-differenced to
    the reference station, against the GNSS Up change at D1's epochs (one
    source for both stations): reported, not graded."""
    if R is None:
        return _labelled("not_testable", "no usable reference station")
    out = {}
    uR = up["stations"].get(R)
    for p in prm.get("validation_stations", []) + prm.get("null_stations", []):
        if p == R:
            continue
        uP = up["stations"].get(p)
        srcs = [s for s in match_src if p in sources.get(s, {}) and R in sources.get(s, {})]
        if uP is None or uR is None or not srcs:
            out[p] = {"why": "no D1 value or no common truth source"}
            continue
        dI = uP["series"] - uR["series"]
        # the first source (in protocol order) that overlaps D1's epochs: the
        # GA workbook ends in July 2019, before A's D1 window, so taking the
        # first source holding both stations gave '0 common epochs' (9 Oct 2026)
        best = None
        for src in srcs:
            eP = match_epochs(sources[src][p], up["days"], prm["match_days"])[0][:, 2]
            eR = match_epochs(sources[src][R], up["days"], prm["match_days"])[0][:, 2]
            dG = eP - eR
            m = np.isfinite(dI) & np.isfinite(dG)
            if best is None or m.sum() > best[2].sum():
                best = (src, dG, m)
            if m.sum() >= 3:
                break
        src, dG, m = best
        if m.sum() < 3:
            out[p] = {"why": f"{int(m.sum())} common epochs (sources tried: {', '.join(srcs)})"}
            continue
        d = (dI - dG)[m]
        out[p] = {"epochs": int(m.sum()), "source": src, "rmse_after_offset_mm": _f(np.sqrt(np.mean((d - d.mean()) ** 2)), 2),
                  "r": _f(pearson(dI[m], dG[m]), 3),
                  "change_insar_mm": _f(dI[m][-1] - dI[m][0], 1), "change_gnss_mm": _f(dG[m][-1] - dG[m][0], 1)}
    return {**_labelled("report_only"), "what": f"D1 Up double-differenced to {R} against GNSS Up ({up['north']})",
            "stations": out, "file": up["path"]}


def _points_row(spec, prm, prod, samples, points_truth, sources, R, flag, comps, stable=None, alias=None) -> dict:
    """Digitised / survey points (CSV): cumulative change from each point's
    first epoch on or after the window start to every later epoch up to the
    window end, against InSAR interpolated to the epoch (SAR dates at most
    survey_max_gap_days apart), both double-differenced to the reference
    station's NGL Up change. Vertical: LOS / cos(incidence), labelled (D1
    Up at the digitised epochs is not wired in yet).

    A point-epoch is 'masked' (honest) only when the point's own cells have
    no reliable value at the SAR dates around it. An epoch outside the run's
    dates, across a SAR gap, without reference GNSS, or where the reference
    station's InSAR is missing is no comparison at all: it is listed, and it
    stays out of the honest-or-flagged share."""
    fl = prm["floor"]
    if R is None or R not in sources.get("ngl", {}):
        return {"what": spec.get("what"), **_labelled("not_testable", "no reference station with NGL truth")}
    Rst = sources["ngl"][R]
    flags = _flags_for(prod, samples, fl, flag[0])
    gap = prm["survey_max_gap_days"]
    sar = prod.days.astype(float)
    wa, wb = (_day(x) if x else None for x in prm["window"])

    def interp(series, day):
        """(value, 'ok' | 'masked' | why there is no value)"""
        j = int(np.searchsorted(sar, day))
        if j < sar.size and sar[j] == day:
            return (float(series[j]), "ok") if np.isfinite(series[j]) else (np.nan, "masked")
        if j == 0 or j >= sar.size:
            return np.nan, "outside the run's dates"
        a, b = j - 1, j
        if sar[b] - sar[a] > gap:
            return np.nan, f"the SAR dates around it are more than {gap} days apart"
        if not (np.isfinite(series[a]) and np.isfinite(series[b])):
            return np.nan, "masked"
        return float(series[a] + (series[b] - series[a]) * (day - sar[a]) / (sar[b] - sar[a])), "ok"

    def ref_up(day):
        d = Rst["days"]
        m = np.abs(d - day) <= prm["match_days"]
        return float(Rst["enu"][m, 2].mean()) if m.any() else np.nan

    Ir = samples[R][fl]["series"] / samples[R][fl]["los"][2]
    sw = prm.get("amend") or {}
    if stable is not None:
        # amendment 2026-10-08/1: the stable placements' vertical (LOS / cos(incidence) at each disc), and
        # the reference station's own GNSS Up sigma for the truth side of the interval
        Sv = stable["series"] / stable["los_up"][None, :]
        rl = prm["stations"][R]
        dist_q = _km_between(rl["lat"], rl["lon"], stable["lat"], stable["lon"])
        s_ref_up = daily_sigma_and_range(Rst, np.array([0.0, 0.0, 1.0]), prm)[0]
    errs, out_c = [], []
    for pt, df in points_truth:
        pid = pt["id"]
        if pid not in samples:
            continue
        u = df[df["component"] == "u"].sort_values("time")
        if not len(u):
            continue
        days = np.array([_day(np.datetime64(t, "D")) for t in u["time"].values])
        vals, sig = u["value_mm"].to_numpy(float), u["sigma_mm"].to_numpy(float)
        keep = np.ones(days.size, bool)
        if wa is not None:
            keep &= days >= wa
        if wb is not None:
            keep &= days <= wb
        days, vals, sig = days[keep], vals[keep], sig[keep]
        if days.size < 2:
            continue
        Ip = samples[pid][fl]["series"] / samples[pid][fl]["los"][2]
        d0 = days[0]
        for k in range(1, days.size):
            tr = (vals[k] - vals[0]) - (ref_up(days[k]) - ref_up(d0))
            (p0, s0), (p1, s1) = interp(Ip, d0), interp(Ip, days[k])
            (r0, q0), (r1, q1) = interp(Ir, d0), interp(Ir, days[k])
            why = next((s for s in (s0, s1) if s not in ("ok", "masked")), None)
            masked = why is None and "masked" in (s0, s1)
            if why is None and not masked and (q0 != "ok" or q1 != "ok"):
                why = f"{R}: " + next(q for q in (q0, q1) if q != "ok")
            if why is None and not math.isfinite(tr):
                why = f"no {R} GNSS within +-{prm['match_days']} d of an epoch"
            ins = (p1 - p0) - (r1 - r0) if why is None and not masked else float("nan")
            rec = {"row": "tahmoor_w_units", "point": pid, "kind": "point_epoch", "date": _iso(days[k]),
                   "since": _iso(d0), "insar": _f(ins, 1), "truth": _f(tr, 1),
                   "error": _f(ins - tr, 1) if math.isfinite(ins) and math.isfinite(tr) else None,
                   "masked": masked, "flagged": flags.get(pid), "truth_sigma": _f(math.hypot(sig[0], sig[k]), 1)}
            if alias is not None and alias.ok and not masked:
                # C-alias: the unit's own interval, as the flag definition says
                rec["flag"] = alias.point(samples[pid][fl]["rows"], samples[pid][fl]["cols"], d0, days[k])
                rec["flagged"] = rec["flag"]["flagged"]
            if why:
                rec["no_comparison"] = why
            elif stable is not None and not masked:
                q = (_interp_cols(Sv, sar, days[k], gap) - _interp_cols(Sv, sar, d0, gap)) - (r1 - r0)
                s_tr = math.sqrt(sig[0] ** 2 + sig[k] ** 2 + 2.0 * s_ref_up ** 2) if math.isfinite(s_ref_up) else None
                _interval_fields(rec, stable_change_sigma(q, dist_q, float(_km_between(rl["lat"], rl["lon"], pt["lat"],
                                                                                         pt["lon"])), sw), s_tr)
            out_c.append(rec)
            if rec["error"] is not None and not rec.get("flagged"):
                errs.append(ins - tr)
    comps.extend(out_c)
    es = error_stats(errs)
    M = {"rmse": es.get("rmse"), "n": es.get("n", 0), "bias": es.get("bias")}
    if not errs:
        g = _labelled("not_testable", "no unflagged point-epoch with values: not testable, never a pass")
    elif not flag[2]:
        g = _labelled("report_only", "computed without the F2 flag (not written yet): reported, not graded")
    else:
        g = grade_row(spec.get("grade"), M)
    return {"what": spec.get("what"), **g, "metrics": M, "comparisons": len(out_c),
            "no_comparison": sum(1 for c in out_c if c.get("no_comparison")),
            "label_vertical": "LOS / cos(incidence): assumes vertical motion"}


def flag_selectivity(stable, comps, alias, layer, limit) -> dict:
    """Amendment 2026-10-08/p23-2: is the flag selective? The share of the
    stable-ground placements (amendment 2026-10-08/1's discs: the stable ring,
    else the reference area) that the flag marks, taken over the SAME
    intervals as the honest-or-flagged row's comparisons: for each comparison
    that is not masked, the share of the placements flagged for its interval
    (a placement is flagged as a point is: at least half of its reliable cells
    carry a flagged level), then the mean over the comparisons (the row's own
    weighting). Comparisons with no interval flag use the whole-record layer.
    Stable ground has nothing to alias, so every flag there is a false alarm.
    'selective' is None when it cannot be measured."""
    out = {"limit": limit, "amendment": "2026-10-08/p23-2",
           "what": "share of the stable-ground placements the flag marks over the comparisons' own intervals"}
    cells = (stable or {}).get("_cells")
    if not stable or cells is None or not np.size(cells.get("rows")):
        out.update(stable_flagged_share=None, selective=None,
                   why=(stable or {}).get("why") or "no stable-ground placement with a reliable cell")
        return out
    R, C, owner = (np.asarray(cells[k]) for k in ("rows", "cols", "owner"))
    n_q = int(owner.max()) + 1 if owner.size else 0
    cache = {}

    def share_for(key):
        if key in cache:
            return cache[key]
        if key == "whole":
            if layer is None:
                cache[key] = None
                return None
            v = layer[R, C].astype(float)
        else:
            v = alias.levels(R, C, key[0], key[1])
        fin = np.isfinite(v)
        hit = fin & np.isin(np.where(fin, v, -1), alias.flagged if alias is not None else ALIAS_FLAGGED)
        n_fin = np.bincount(owner, weights=fin.astype(float), minlength=n_q)
        n_hit = np.bincount(owner, weights=hit.astype(float), minlength=n_q)
        ev = n_fin > 0
        cache[key] = (float(np.mean(n_hit[ev] / n_fin[ev] >= 0.5)), int(ev.sum())) if ev.any() else None
        return cache[key]

    vals, n_eval, by_kind, basis = [], [], {}, {"interval": 0, "whole_record": 0}
    for c in comps:
        if c.get("masked") or c.get("truth") is None or c.get("no_comparison"):
            continue
        f = c.get("flag") or {}
        iv = f.get("interval") if f.get("basis") == "interval" else None
        key = (_day(iv[0]), _day(iv[1])) if iv and alias is not None and getattr(alias, "ok", False) else "whole"
        r = share_for(key)
        if r is None:
            continue
        basis["interval" if key != "whole" else "whole_record"] += 1
        vals.append(r[0])
        n_eval.append(r[1])
        by_kind.setdefault(c.get("kind", "?"), []).append(r[0])
    if not vals:
        out.update(stable_flagged_share=None, selective=None,
                   why="no comparison with a flag to measure against (no per-pair record and no flag layer)")
        return out
    sh = float(np.mean(vals))
    out.update(stable_flagged_share=_f(sh, 4), selective=bool(sh <= float(limit)),
               comparisons=len(vals), intervals=len(cache), placements_evaluated_median=int(np.median(n_eval)),
               interval_share_min=_f(min(vals), 4), interval_share_max=_f(max(vals), 4), basis=basis,
               by_kind={k: {"n": len(v), "stable_flagged_share": _f(float(np.mean(v)), 4)} for k, v in by_kind.items()},
               stable_ground=stable.get("how"))
    return out


def _unflagged_row(spec, comps, prod, samples, fl, flag, alias=None, stable=None, sw=None, run=None) -> dict:
    """Share of all truth comparisons whose error exceeds max(30 mm, 20% of
    the truth) and that are neither masked nor flagged; with the shares
    flagged, masked, and the flag's false alarms beside it. A comparison
    carries its own interval's flag when the per-pair aliasing record was
    read (C-alias, `alias`); otherwise the point's whole-record flag.
    Amendment 2026-10-08/p23-2: the row can pass only if the flag is
    selective (flag_selectivity on the stable ground <= the limit); else it
    reads 'flag not selective' (fail). Amendment 2026-10-08/p23-6: on a run
    whose honesty cannot be assessed (A2, DISP-S1) the row says so and does
    not grade."""
    sw = sw or {}
    layer, what, grades = flag
    flags = _flags_for(prod, samples, fl, layer)
    # a comparison with no truth, or none made (the reference station's InSAR missing, an epoch
    # outside the run's dates): not a point that was honest, so not in the denominator
    used = [c for c in comps if c.get("truth") is not None and not c.get("no_comparison")]
    n = len(used)
    if not n:
        return {"what": spec.get("what"), **_labelled("not_testable", "no truth comparison was made"),
                "metrics": {"n": 0, "not_compared": len(comps)}}
    fails = flagged = masked = false_alarm = 0
    for c in used:
        fg = bool(flags.get(c.get("point"))) if c.get("flagged") is None else bool(c["flagged"])
        c["flagged"] = fg
        tr = c.get("truth")
        bad = (c.get("error") is not None and tr is not None and abs(c["error"]) > max(30.0, 0.2 * abs(tr)))
        if c.get("masked"):
            masked += 1
        elif fg:
            flagged += 1
            if c.get("error") is not None and tr is not None and not bad:
                false_alarm += 1
        elif bad:
            fails += 1
    M = {"rate": fails / n, "n": n, "unflagged_failures": fails, "share_flagged": flagged / n,
         "share_masked": masked / n, "flag_false_alarm_share": (false_alarm / flagged) if flagged else None,
         "not_compared": len(comps) - n}
    g = grade_row(spec.get("grade"), M) if grades else _labelled(
        "report_only", f"computed with {what}: this version does not grade (protocol A)")
    sel = None
    lim = sw.get("flag_selectivity_max_stable_share")
    if lim is not None and grades:
        sel = flag_selectivity(stable, used, alias, layer, float(lim))
        if sel.get("selective") is False:
            g = {**_labelled("fail", f"the flag marks {100 * sel['stable_flagged_share']:.1f}% of the stable ground over "
                                     f"the comparisons' intervals, above the {100 * float(lim):.0f}% limit: a flag that "
                                     "covers stable ground as readily as the truth points cannot show honesty "
                                     "(amendment 2026-10-08/p23-2)"),
                 "label": "flag not selective", "rule": g.get("rule"), "rate_grade": g.get("grade")}
        elif sel.get("selective") is None and g.get("grade") in ("pass", "marginal"):
            g = {**_labelled("not_testable", "the flag's selectivity could not be measured on stable ground ("
                             + str(sel.get("why")) + "): the row can pass only with a selective flag (amendment "
                                                     "2026-10-08/p23-2)"), "rate_grade": g.get("grade")}
    honesty = (sw.get("accuracy_only_verdict") or {}).get(str(run)) if run else None
    if honesty:
        g = {**_labelled("not_testable", honesty), "label": honesty,
             "amendment": (sw.get("_by") or {}).get("accuracy_only_verdict")}
    n_int = sum(1 for c in used if (c.get("flag") or {}).get("basis") == "interval")
    basis = {"interval": n_int, "whole_record_or_none": n - n_int,
             "note": ("each comparison flagged for its own interval (the per-pair aliasing record)" if n_int == n else
                      "the point's whole-record flag, which flags more than any one interval: " +
                      (alias.why if alias is not None and alias.why else "no per-pair aliasing record was read")
                      if n_int == 0 else "mixed: the comparisons without an interval flag use the whole-record flag")}
    by_kind = {}
    for c in used:
        k = by_kind.setdefault(c.get("kind", "?"), {"n": 0, "masked": 0, "flagged": 0, "unflagged_failures": 0})
        k["n"] += 1
        tr = c.get("truth")
        bad = (c.get("error") is not None and tr is not None and abs(c["error"]) > max(30.0, 0.2 * abs(tr)))
        if c.get("masked"):
            k["masked"] += 1
        elif c.get("flagged"):
            k["flagged"] += 1
        elif bad:
            k["unflagged_failures"] += 1
    out = {"what": spec.get("what"), **g, "metrics": M, "flag_layer": what, "flag_basis": basis, "by_kind": by_kind}
    if sel is not None:
        out["flag_selectivity"] = sel
        if sel.get("stable_flagged_share") is not None and M["share_flagged"] is not None:
            # report only: how much more the flag marks the truth points than stable ground
            sel["truth_flagged_share"] = _f(M["share_flagged"], 4)
    return out


# =============================================================== verdict

def route_verdict(P, rows: dict, comparisons=None, run=None) -> dict:
    """Grades of the protocol's graded rows and the route verdict its
    `route_trustworthy_*` rule names (rows named outside brackets must all
    pass). Amendment 2026-10-08/p23-6: a run named in the amendment key
    accuracy_only_verdict (A2, DISP-S1) gets the rule read without its
    honest-or-flagged part, and without the rows the protocol does not run
    on it, labelled with the amendment's text: an accuracy verdict, never a
    verdict on honesty."""
    V = P.get("verdict") or {}
    graded = list(V.get("rows_graded") or [])
    table = {r: (rows.get(r) or {}).get("grade", "not_computed") for r in graded}
    key = next((k for k in V if str(k).startswith("route_trustworthy")), None)
    text = re.sub(r"\([^()]*\)", "", str(V.get(key, "")))
    needed = [r for r in (P.get("comparisons") or {}) if re.search(r"\b" + re.escape(r) + r"\b", text)]
    sw = amendment_switches(P)
    honesty = (sw.get("accuracy_only_verdict") or {}).get(str(run)) if run else None
    left_out = []
    if honesty:
        C_ = P.get("comparisons") or {}
        left_out = [r for r in needed if r == "unflagged_failure_rate" or _not_this_run(C_.get(r), run)]
        needed = [r for r in needed if r not in left_out]
    extra = {}
    if P.get("protocol") == "A" or "bias = |mean signed error|" in str(V.get(key, "")):
        extra = _route_extras_A(V.get(key, ""), comparisons or [], sw)
    states = [table.get(r, (rows.get(r) or {}).get("grade", "not_computed")) for r in needed] + \
        [e["grade"] for e in extra.values()]
    if any(s == "fail" for s in states):
        g, txt = "fail", str(V.get("fail", "fails: blocks this regime from client reports"))
    elif any(s in ("not_computed", "not_testable", "report_only") for s in states) or not states:
        missing = [r for r in needed if table.get(r, (rows.get(r) or {}).get("grade")) not in ("pass", "marginal", "fail")]
        missing += [k for k, e in extra.items() if e["grade"] not in ("pass", "marginal", "fail")]
        g, txt = "not_reached", "no route verdict: " + (", ".join(missing) or "no key row") + " not graded"
    elif all(s == "pass" for s in states):
        g, txt = "pass", "trustworthy for this regime at this site (the route rule's rows all pass)"
    else:
        g, txt = "marginal", str(V.get("marginal", "usable with stated caveats"))
    out = {"rows": table, "route_rule": str(V.get(key, "")).strip(), "route_rows": needed, "route_extras": extra,
           "route": {"grade": g, "text": txt}}
    if honesty:
        parts = ", ".join(needed + list(extra)) or "none"
        out["route"] = {"grade": g, "scope": "accuracy only", "honesty": honesty,
                        "amendment": (sw.get("_by") or {}).get("accuracy_only_verdict"),
                        "text": (f"accuracy only ({parts}): "
                                 + {"pass": "the accuracy parts all pass",
                                    "marginal": "usable with stated caveats on accuracy",
                                    "fail": "fails on accuracy",
                                    "not_reached": txt}.get(g, txt)
                                 + f"; {honesty}; rows left out: {', '.join(left_out) or 'none'}")}
        out["route_rows_left_out"] = left_out
    return out


def _route_extras_A(rule: str, comparisons, sw=None) -> dict:
    """Protocol A's own reading of bias and uncertainty calibration for event
    quantities: |mean signed error| over the CA19 event changes and the
    digitised-unit changes <= the stated mm, and >= 90% of those errors
    inside the stated 95% interval. As first fixed the deliverables state no
    uncertainty for a displacement change, so that part is 'not stated';
    under amendment 2026-10-08/1 the interval is the harness's, from stable
    ground, and the part passes at >= the amendment's share (90%), else fails."""
    sw = sw or {}
    lim = _num(r"<=\s*([\d.]+)\s*mm", rule, 30.0)
    used = [c for c in comparisons
            if c.get("kind") in ("event", "point_epoch") and not c.get("masked") and c.get("error") is not None]
    errs = [c["error"] for c in used]
    out = {}
    if errs:
        b = float(np.mean(errs))
        out["bias"] = {"grade": "pass" if abs(b) <= lim else "fail", "value_mm": _f(b, 1), "limit_mm": lim,
                       "n": len(errs)}
    else:
        out["bias"] = {"grade": "not_testable", "why": "no event or digitised comparison with a value"}
    if sw.get("event_change_sigma") == "stable_ground":
        stated = [c for c in used if c.get("interval_95_mm") is not None]
        need = float(sw.get("uncertainty_share_pass", 0.90))
        if not stated:
            why = sorted({str(c.get("interval_why") or "no interval computed") for c in used}) or ["no comparison"]
            out["uncertainty_calibration"] = {"grade": "not_testable", "amendment": "2026-10-08/1",
                                              "why": "no event or digitised comparison has a stable-ground interval: "
                                                     + "; ".join(why)}
        else:
            n_in = sum(1 for c in stated if c.get("inside_interval"))
            share = n_in / len(stated)
            out["uncertainty_calibration"] = {
                "grade": "pass" if share >= need else "fail", "share_inside": _f(share, 4), "inside": n_in,
                "n": len(stated), "not_stated": len(used) - len(stated), "pass_at": need, "amendment": "2026-10-08/1",
                "interval": "1.96 x sqrt(s_insar^2 + s_truth^2), s_insar from the stable ground (amendment 2026-10-08/1)"}
    else:
        out["uncertainty_calibration"] = {"grade": "not_testable",
                                          "why": "not stated: the deliverables state no uncertainty for a displacement change"}
    return out


# =============================================================== outputs

def _clean(o):
    """Drop private keys (leading underscore) and arrays meant for plots."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_") and k not in ("series", "plane_series", "plane_joint_series", "rows_cells")}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def _ci_text(ci):
    if not ci or ci.get("lo") is None:
        return "no interval"
    return f"{ci['lo']:.3g} to {ci['hi']:.3g}"


def _row_value_text(name, row):
    M = row.get("metrics") or {}
    ci = row.get("ci") or {}
    for key, unit in (("rmse", "mm/yr" if name.startswith("rate") else "mm"), ("median_rmse", "mm"),
                      ("abs_bias", "mm/yr"), ("slope", ""), ("r", ""), ("share_inside", ""),
                      ("abs_err", "mm"), ("rate", ""), ("ratio", ""), ("share_beyond", "")):
        v = M.get(key)
        if v is None:
            continue
        if key == "abs_bias":
            return f"bias {M.get('bias'):+.2f} mm/yr [95% CI {_ci_text(ci.get('bias'))}]"
        if key == "share_inside":
            return f"{M.get('n', 0) - (M.get('n_outside') or 0)} of {M.get('n')} inside ({100 * v:.0f}%)"
        if key == "abs_err":
            return f"error {M.get('err'):+.1f} mm against truth {M.get('truth'):+.1f} mm"
        if key == "ratio":
            return (f"ratio {v:.2f}" + (f" (lowest of {M['n']} panel-lines)" if M.get("n") else "")
                    + (f", peak {M['peak_mm']:+.0f} mm" if M.get("peak_mm") is not None and not M.get("n") else ""))
        c = ci.get(key) or ci.get("rmse")
        return f"{key.replace('_', ' ')} {v:.3g} {unit} [95% CI {_ci_text(c)}]".replace("  ", " ")
    return "-"


def write_markdown(path, res: dict) -> None:
    pr, pd_ = res["protocol"], res["product"]
    v = res["verdict"]
    rel_word = ("mean pair coherence (NISAR chain)" if str(pd_.get("reliability_quantity", "")).startswith("mean")
                else "temporal coherence")
    L = [f"# Validation: protocol {pr['id']}" + (f", run {pr['run']}" if pr.get("run") else ""),
         "",
         f"- Protocol: `{pr['path']}` (version {pr['version']}). {pr['fixed']['text'][:1].upper()}{pr['fixed']['text'][1:]}.",
         f"- Product: `{pd_['timeseries']}`, {pd_['dates']} dates, {pd_['first']} to {pd_['last']}.",
         f"- Truth: {', '.join(sorted(res['truth']['stations'])) or 'none loaded'}"
         + (f"; not loaded: {', '.join(f'{k} ({w})' for k, w in res['truth']['not_loaded'].items())}"
            if res["truth"]["not_loaded"] else "") + ".",
         f"- Sampling: median of reliable cells within {res['parameters']['radius_m']:.0f} m, {rel_word} "
         f">= {res['parameters']['floor']}, at least {res['parameters']['min_cells']} cells and "
         f"{res['parameters']['min_independent_pixels']} independent radar pixels per date. GNSS: mean of the days "
         f"within +-{res['parameters']['match_days']} days of each SAR date.",
         "", "## Verdict", "",
         f"**{GRADE_TEXT.get(v['route']['grade'], v['route']['grade'])}**: {v['route']['text']}", "",
         "| row | result | grade |", "|---|---|---|"]
    if pr.get("amendments"):
        ft = pr.get("fixed_text") or {}
        L.insert(3, f"- Amendments: {', '.join(str(a.get('id')) for a in pr['amendments'] if isinstance(a, dict))}, "
                    f"dated 8 Oct 2026, before any result. Fixed text: {ft.get('text', 'not checked')}.")
    rows = res["rows"]
    for name in v["rows"]:
        r = rows.get(name, {})
        txt = _row_value_text(name, r) if r.get("metrics") else "-"
        if txt == "-" and r.get("why"):
            txt = r["why"]
        if r.get("amendment"):
            txt += f" (amendment {r['amendment']})"
        fs = r.get("flag_selectivity")
        if fs:
            txt += ("; flag on stable ground: " + (f"{100 * fs['stable_flagged_share']:.1f}%" if fs.get(
                "stable_flagged_share") is not None else f"not measured ({fs.get('why')})")
                    + f" (limit {100 * fs['limit']:.0f}%, amendment {fs['amendment']})")
        L.append(f"| {name} | {txt} | {r.get('label', GRADE_TEXT['not_computed'])} |")
    for k, e in (v.get("route_extras") or {}).items():
        val = (f"{e['inside']} of {e['n']} inside ({100 * e['share_inside']:.0f}%)" if "share_inside" in e
               else f"{e['value_mm']:+.1f} mm (limit {e['limit_mm']:g})" if "value_mm" in e else e.get("why", "-"))
        L.append(f"| route rule: {k} | {val}{' (amendment ' + e['amendment'] + ')' if e.get('amendment') else ''} | "
                 f"{GRADE_TEXT.get(e['grade'], e['grade'])} |")
    pc = rows.get("plane_check")
    if pc and pc.get("rows"):
        L += ["", f"Plane check (amendment {pc.get('amendment')}): {pc.get('label')}. The same rows in the joint offset "
                  "+ plane datum, graded against the same thresholds; not part of the route verdict.", "",
              "| row | result | grade |", "|---|---|---|"]
        for n, s in pc["rows"].items():
            L.append(f"| {n} | {_row_value_text(n, s) if s.get('metrics') else '-'} | {s.get('label')} |")
    v1 = res.get("version_1")
    if v1:
        g1 = v1["verdict"]["route"]
        L += ["", f"Under the protocol as first fixed (amendments set aside): **{GRADE_TEXT.get(g1['grade'], g1['grade'])}**: "
                  f"{g1['text']}"]
        diff = [n for n in v["rows"] if (v1["rows"].get(n) or {}).get("grade") != (rows.get(n) or {}).get("grade")]
        if diff:
            L.append("Rows graded differently there: " + "; ".join(
                f"{n} {(v1['rows'].get(n) or {}).get('label', '-')}" for n in diff) + ".")
    others = [n for n in rows if n not in v["rows"]]
    if others:
        L += ["", "Reported, not graded:", ""]
        for n in others:
            r = rows[n]
            L.append(f"- {n}: {r.get('label')}" + (f" ({r['why']})" if r.get("why") else ""))
    sens = res.get("sensitivity") or {}
    if sens:
        L += ["", "## Coherence-floor sensitivity (reported beside the primary, never replacing it)", "",
              "| floor | " + " | ".join(res["verdict"]["rows"]) + " |",
              "|---|" + "---|" * len(res["verdict"]["rows"])]
        for fl, rr in sens.items():
            L.append(f"| {fl} | " + " | ".join((rr.get(n) or {}).get("label", "-") for n in res["verdict"]["rows"]) + " |")
    L += ["", "## Read this first", "", str(res.get("power_statement") or "").strip() or
          "Few stations: every figure carries a bootstrap interval; a pass on the point estimate is never reported alone."]
    cmr = (rows.get("york_slope") or {}).get("common_mode_removed")
    amended_slope = (rows.get("york_slope") or {}).get("amendment") == "2026-10-08/2"
    if cmr and cmr.get("n"):
        L += ["", ("The rate-based York slope and rate correlation (reported, not graded under amendment "
                   "2026-10-08/2; the graded slope is on the per-station changes) " if amended_slope else
                   "York slope and rate correlation ") +
              "pool water-year rates in the leave-one-out datum, which brings "
              "the whole box's motion from the other stations' GNSS into the InSAR, so both read close to 1 even "
              "when the InSAR gets the motion between stations wrong. With each water year's mean removed "
              f"(not graded): York slope {cmr.get('york_slope')}, r {cmr.get('r')}, over {cmr['n']} rates."]
    if res.get("warnings"):
        L += ["", "## Warnings", ""] + [f"- {w}" for w in res["warnings"]]
    L += ["", f"Full detail: `validation.json`; per-date values: `validation_epochs.csv`. Written {res['harness']['run_at']}."]
    Path(path).write_text("\n".join(L) + "\n")


def write_epochs_csv(path, table) -> None:
    if not table:
        Path(path).write_text("no matched epochs\n")
        return
    from .areas import formula_safe
    keys = list(dict.fromkeys(k for r in table for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in table:
            # station and point ids come from the truth files: never a
            # formula in a spreadsheet (16 Oct 2026, readiness review SEC-05)
            w.writerow({k: formula_safe(v) for k, v in r.items()})


# colours: the dataviz reference palette, light mode (InSAR = slot 1, truth = slot 2)
_INK, _INK2, _GRID, _SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
_C_INSAR, _C_TRUTH = "#2a78d6", "#eb6834"


def _axes_style(ax):
    ax.set_facecolor(_SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_INK2)
    ax.tick_params(colors=_INK2, labelsize=8)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_scatter(path, x, y, sx=None, sy=None, labels=None, xlabel="", ylabel="", title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = np.asarray(x, float), np.asarray(y, float)
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150, facecolor=_SURF)
    _axes_style(ax)
    lo = float(np.nanmin(np.r_[x, y])) if x.size else -1
    hi = float(np.nanmax(np.r_[x, y])) if x.size else 1
    pad = 0.08 * (hi - lo or 1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=_INK2, lw=1, ls="--", label="1:1")
    if sx is not None or sy is not None:
        ax.errorbar(x, y, xerr=sx, yerr=sy, fmt="none", ecolor=_INK2, elinewidth=0.8, alpha=0.6)
    ax.scatter(x, y, s=36, color=_C_INSAR, edgecolor=_SURF, linewidth=1.5, zorder=3, label="InSAR against truth")
    if labels is not None:
        seen = set()                     # selective labels: each name once, at its first point
        for xi, yi, t in zip(x, y, labels):
            if t in seen or len(seen) >= 12 or not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            seen.add(t)
            ax.annotate(str(t), (xi, yi), xytext=(5, -10), textcoords="offset points", fontsize=7, color=_INK2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(xlabel, color=_INK, fontsize=9)
    ax.set_ylabel(ylabel, color=_INK, fontsize=9)
    ax.set_title(title, color=_INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=_INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURF)
    plt.close(fig)


def plot_series(path, panels, title=""):
    """panels: [(name, dates (YYYYMMDD), insar, truth, subtitle)] as small multiples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = max(1, len(panels))
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.1 * n + 0.6), dpi=150, sharex=True, facecolor=_SURF, squeeze=False)
    for ax, (name, dates, ins, tru, sub) in zip(axes[:, 0], panels):
        _axes_style(ax)
        t = np.array([np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:8]}") for d in dates])
        ax.plot(t, tru, color=_C_TRUTH, lw=2, marker="o", ms=3, label="GNSS (LOS)")
        ax.plot(t, ins, color=_C_INSAR, lw=2, marker="o", ms=3, label="InSAR (LOS)")
        ax.set_ylabel("mm", color=_INK2, fontsize=8)
        ax.set_title(f"{name}  {sub}", color=_INK, fontsize=9, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=_INK2, ncol=2, loc="upper right")
    if title:
        fig.suptitle(title, color=_INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURF)
    plt.close(fig)


def write_plots(out_dir, res, raw) -> list[str]:
    out = []
    try:
        fl = res["parameters"]["floor"]
        if raw["design"] == "network":
            comp = raw["by_floor"][fl]["los"]
            rr = comp["rates"]["rows"]
            if rr:
                p = Path(out_dir) / "rates_scatter.png"
                plot_scatter(p, [r["gnss"] for r in rr], [r["insar"] for r in rr],
                             [r["s_gnss"] for r in rr], [r["s_insar"] for r in rr],
                             [r["station"] for r in rr],
                             "GNSS LOS rate (mm/yr)", "InSAR LOS rate, leave-one-out absolute (mm/yr)",
                             "Rates: InSAR against GNSS")
                out.append(p.name)
            dd = comp["dd"]["pairs"]
            xs = [(p_["pair"], lab, p_["gnss"][lab], p_["insar"][lab]) for p_ in dd for lab in p_["error"]]
            if xs:
                p = Path(out_dir) / "dd_rates_scatter.png"
                plot_scatter(p, [x[2] for x in xs], [x[3] for x in xs], labels=[x[0] for x in xs],
                             xlabel="GNSS LOS rate difference (mm/yr)", ylabel="InSAR LOS rate difference (mm/yr)",
                             title="Station-pair double differences")
                out.append(p.name)
            panels = []
            for s, A in comp["loo"]["series"].items():
                G = raw["by_floor"][fl]["_G"][s]
                m = np.isfinite(A) & np.isfinite(G)
                if m.sum() >= 3:
                    off = float((A[m] - G[m]).mean())
                    panels.append((s, [d for d, k in zip(raw["dates"], m) if k], A[m] - off, G[m],
                                   f"leave-one-out absolute, offset {off:+.1f} mm removed"))
            if panels:
                p = Path(out_dir) / "series_stations.png"
                plot_series(p, panels, "LOS time series at each station")
                out.append(p.name)
        else:
            rows = raw["by_floor"][fl]["rows"]
            ev = [r["comparison"] for r in rows.values() if r.get("comparison") and r["comparison"].get("error") is not None]
            if ev:
                p = Path(out_dir) / "events_scatter.png"
                plot_scatter(p, [e["truth"] for e in ev], [e["insar"] for e in ev], labels=[e["row"] for e in ev],
                             xlabel="GNSS LOS change, double-differenced (mm)",
                             ylabel="InSAR LOS change, double-differenced (mm)", title="Event changes")
                out.append(p.name)
            panels = []
            for name, r in rows.items():
                s = r.get("_series")
                if s:
                    panels.append((name, s["dates"], np.array(s["insar"]), np.array(s["truth"]),
                                   f"double difference to {raw['by_floor'][fl]['reference_choice'].get('chosen')}"))
            if panels:
                p = Path(out_dir) / "series_points.png"
                plot_series(p, panels, "LOS time series, double-differenced")
                out.append(p.name)
    except Exception as exc:                                  # noqa: BLE001 - plots never stop the numbers
        log.warning("Plots not written (%s)", exc)
    return out


# =============================================================== the run

def product_reference(cfg, prm) -> tuple[dict, list[str]]:
    """What the product's own relative figures were made relative to
    (config reference_mode, significance.reference_mode) and, in
    gnss_absolute mode, the stations its datum was fitted to
    (deliverables/reference_datum.json). The harness compares the time
    series, which stay on the run's reference pixel in every mode, so its
    differences do not depend on this; but the stated rate uncertainty the
    uncertainty-calibration rows read (uncertainty_model.json) gains the
    datum's leave-one-out error in that mode. A station the protocol scores
    must not have set that datum (protocols: no validation station anchors
    or references the InSAR): warned when it did. Returns (record, warnings)."""
    warn = []
    try:
        mode = _significance().reference_mode(cfg)
    except ValueError as exc:
        return {"reference_mode": str(getattr(cfg, "reference_mode", "")), "problem": str(exc)}, \
            [f"The product's reference_mode is not understood ({exc})"]
    rec = {"reference_mode": mode}
    if mode == "stable_polygon":
        rec["stable_polygon"] = str(getattr(cfg, "reference_stable_geojson", "") or "")
    elif mode == "gnss_absolute":
        named = [str(x).strip().upper() for x in (getattr(cfg, "reference_gnss_stations", None) or [])]
        rec["config_stations"] = named
        used = []
        rp = Path(cfg.project_dir) / "deliverables" / "reference_datum.json"
        if rp.exists():
            try:
                rd = json.loads(rp.read_text())
                used = [str(m.get("id", "")).upper() for m in rd.get("stations", []) if m.get("in_fit")]
                rec.update(record=str(rp), available=rd.get("available"), datum_stations=used)
            except (ValueError, OSError) as exc:
                rec["record_problem"] = f"{rp.name} could not be read ({exc})"
        else:
            rec["record_problem"] = f"no {rp.name} in deliverables/"
        scored = set(prm.get("validation_stations", []) + prm.get("null_stations", [])
                     + prm.get("calibration_pool", []) + prm.get("rate_stations", []) + prm.get("series_stations", []))
        clash = sorted(scored & set(used or named))
        rec["scored_stations_in_datum"] = clash
        if clash:
            warn.append(f"The product was made with reference_mode gnss_absolute, its datum fitted to "
                        f"{', '.join(clash)}, which this protocol scores: the stated rate uncertainty the "
                        f"uncertainty-calibration rows read was calibrated on the truth it is compared with. "
                        f"The series comparisons are unaffected (they stay on the reference pixel). Re-run the "
                        f"products with reference_mode farmland, or with datum stations the protocol does not score")
    return rec, warn


def _find_run(P, config_path) -> str | None:
    """The protocol's run whose config has this file name: runs.<id>.config,
    or, for a run whose config was written after the protocol was fixed, the
    run_configs record of its amendments (2026-10-08/configs)."""
    name = Path(config_path).name if config_path else ""
    if not name:
        return None
    for rid, r in (P.get("runs") or {}).items():
        if isinstance(r, dict) and Path(str(r.get("config", ""))).name == name:
            return str(rid)
    for rid, c in (amendment_switches(P).get("run_configs") or {}).items():
        if name in [Path(str(x)).name for x in (c if isinstance(c, list) else [c])] and rid in (P.get("runs") or {}):
            return str(rid)
    return None


def _compare(P, prm, cfg, prod, sources, samples, floors, pts_truth, flag, up, up_samples, run, stable,
             alias=None, events_in=None):
    """Every comparison and row of the protocol P (as given: amended, or as
    first fixed) on the sampled product and the loaded truth, and its route
    verdict. Returns (raw, rows, sensitivity rows by floor, verdict)."""
    if prm["design"] == "network":
        raw = run_network(P, prm, prod, sources["ngl"], samples, floors, up_samples)
        prim = raw["by_floor"][prm["floor"]]
        rows = dict(prim["rows"])
        rows_extra = {}
        vrows = prim.get("vertical_assumed_rows") or {}
        rows_extra["vertical_secondary"] = {
            "what": ((P.get("comparisons") or {}).get("vertical_secondary") or {}).get("what"),
            **_labelled("report_only"),
            "assumes_vertical": {k: {"metrics": v.get("metrics"), "ci": v.get("ci")} for k, v in vrows.items()},
            "label": GRADE_TEXT["report_only"],
            "note": "LOS / cos(incidence) against GNSS Up, labelled 'assumes vertical motion'"}
        if prim.get("up_d1_rows"):
            rows_extra["vertical_secondary"]["d1_up"] = {k: {"metrics": v.get("metrics"), "ci": v.get("ci")}
                                                         for k, v in prim["up_d1_rows"].items()}
            rows_extra["vertical_secondary"]["d1_file"] = up_samples["path"]
        C = P.get("comparisons") or {}
        for name, why in (("tre_vertical", "needs the DWR/TRE-Altamira rasters (no truth adapter for them yet)"),
                          ("residual_variogram", "needs a stable-ground polygon; the protocol expects none in this box"),
                          ("nisar", "not a NISAR run" if getattr(cfg, "backend", "") != "nisar"
                           else "the NISAR rows need M1/M2 (PLAN phase 3)"),
                          ("disp_s1", "run the harness on the DISP-S1 run's own config"),
                          ("vertical_only_bias", "needs the D1 Up product: pass --up" if up is None else "")):
            if name in C and why:
                rows_extra[name] = {"what": (C[name] or {}).get("what") if isinstance(C[name], dict) else None,
                                    **_labelled("not_computed", why)}
        if up is not None and "vertical_only_bias" in C:
            vals = {}
            lo, hi = int(up_samples["days"][0]), int(up_samples["days"][-1])
            for s, u in up_samples["stations"].items():
                if s not in samples or u is None:
                    continue
                smp = samples[s][prm["floor"]]
                vi = smp["series"] / smp["los"][2]
                m = np.isfinite(vi) & (prod.days >= lo) & (prod.days <= hi)
                mu = np.isfinite(u["series"])
                if m.sum() >= 4 and mu.sum() >= 4:
                    om_v = model_rates(prod.days[m], vi[m], prm)["overlap_mean"]
                    om_u = model_rates(up_samples["days"][mu], u["series"][mu], prm)["overlap_mean"]
                    if om_v and om_u:
                        vals[s] = {"los_over_cos_rate": _f(om_v["rate"], 2), "d1_up_rate": _f(om_u["rate"], 2),
                                   "difference": _f(om_v["rate"] - om_u["rate"], 2),
                                   "relative_to_up": _f((om_v["rate"] - om_u["rate"]) / abs(om_u["rate"]), 3)
                                   if om_u["rate"] else None}
            rows_extra["vertical_only_bias"] = {"what": C["vertical_only_bias"].get("what"),
                                                **_labelled("report_only"), "stations": vals,
                                                "window": [_iso(lo), _iso(hi)]}
        dup = {}
        for sid, e in ((P.get("truth") or {}).get("stations") or {}).items():
            if isinstance(e, dict) and "duplicate_check" in _roles(e):
                partner = _stations_named(e.get("why"), {s: 1 for s in ((P.get("truth") or {}).get("stations") or {})})
                partner = [p for p in partner if p != sid]
                dup[sid] = {"partner": partner[0] if partner else None}
        if dup:
            for sid, d in dup.items():
                a, b = sources["ngl"].get(sid), sources["ngl"].get(d["partner"])
                if a is None or b is None:
                    d["why"] = "series not loaded"
                    continue
                common, ia, ib = np.intersect1d(a["days"], b["days"], return_indices=True)
                diff = a["enu"][ia] - b["enu"][ib]
                d.update(days=int(common.size),
                         robust_sd_mm={c: _f(_robust_sigma(diff[:, j] - np.median(diff[:, j])), 2) for j, c in enumerate("enu")})
            rows_extra["duplicate_check"] = {**_labelled("report_only"), "what": "GNSS-only noise check of a duplicate site",
                                             "pairs": dup}
        rows.update(rows_extra)
        raw["dates"] = prod.dates
        sens = {str(fl): raw["by_floor"][fl]["rows"] for fl in floors if fl != prm["floor"]}
    else:
        raw = run_reference(P, prm, prod, sources, samples, floors, pts_truth, flag, up_samples, stable=stable,
                            alias=alias, events_in=events_in)
        prim = raw["by_floor"][prm["floor"]]
        rows = dict(prim["rows"])
        sens = {str(fl): raw["by_floor"][fl]["rows"] for fl in floors if fl != prm["floor"]}
        for k in [k for k in prim if str(k).startswith("reference_") and k != "reference_choice"
                  and not str(k).endswith("_comparisons")]:
            rows[f"sensitivity_{k}"] = {**_labelled("report_only"), "what": f"the same rows with {k[10:]} as the reference",
                                        "rows": {n: {"label": r.get("label"), "metrics": r.get("metrics")}
                                                 for n, r in prim[k].items()}}

    # rows the protocol does not run on this run (its `runs:` list): said so, never graded
    for rr in [rows] + list(sens.values()):
        for name in list(rr):
            why_run = _not_this_run((P.get("comparisons") or {}).get(name), run)
            if why_run and isinstance(rr[name], dict):
                before = rr[name].get("why")
                rr[name] = {**rr[name], **_labelled("not_testable", why_run + (f" (and: {before})" if before else ""))}
    verdict = route_verdict(P, rows, raw["by_floor"][prm["floor"]].get("comparisons"), run=run)
    return raw, rows, sens, verdict


def validate(cfg, protocol_path, out_dir, run=None, cache_dir=None, ga_xlsx=None, points=None, up=None,
             sensitivity=True, epoch_coverage=True, gt=None, config_path=None, outlines=None, lines=None) -> dict:
    """Run the protocol on one run's products; write the outputs; return the result.
    `outlines`: the digitised panel outlines (GeoJSON, PLAN T1) for protocol A's
    stable ring (amendment 2026-10-08/1 places its stable discs there; without
    them, on the ground more than deramp_buffer_m from the run's deramp polygon)
    and the Tahmoor West peak (W1-W4). `lines`: the digitised end-of-panel
    survey lines (GeoJSON) for the Appin peak ratio. Without them the config's
    event_polygons / event_lines are used when set."""
    P = load_protocol(protocol_path)
    run = run or _find_run(P, config_path)
    prm = protocol_params(P, run, getattr(cfg, "backend", None))
    # truth points: --points, else the config's validation_points (the products stage passes no --points)
    points_from = "--points" if points else None
    if not points and str(getattr(cfg, "validation_points", "") or "").strip():
        points, points_from = str(cfg.validation_points).strip(), "the config's validation_points"
    if prm["design"] is None:
        raise ValueError("The protocol names neither a reference station (reference.primary.station) nor "
                         "network modes (reference.modes.double_difference): the harness cannot tell which design")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else out_dir / "gnss_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gt = gt or _ground_truth()
    warn = []
    if run is None and (P.get("runs") or {}):
        warn.append(f"This config is not one of the protocol's runs ({', '.join(map(str, P['runs']))}; neither "
                    "runs.<id>.config nor an amendment's run_configs names it): every row is computed, over the "
                    "protocol's whole window. These are not the protocol's results for any run.")
    fixed = fixed_hash_status(protocol_path)
    if fixed["status"] == "changed":
        warn.append("The protocol file differs from the one fixed on 8 Oct 2026: " + fixed["text"])
    if P.get("_inherited"):
        # B-2 takes its sampling, matching and statistics sections from B-1: an edited B-1 changes B-2 too
        pfix = fixed_hash_status(P["_inherited"]["file"])
        P["_inherited"]["fixed"] = {k: pfix[k] for k in ("status", "text")}
        if pfix["status"] == "changed":
            warn.append(f"The protocol it inherits from ({Path(P['_inherited']['file']).name}) differs from the one "
                        f"fixed on 8 Oct 2026: " + pfix["text"])
    fixed_text = None
    if P.get("amendments"):
        fixed_text = fixed_text_status(protocol_path)
        warn.append(f"The protocol has {len(P['amendments'])} dated amendment(s) "
                    f"({', '.join(str(a.get('id')) for a in P['amendments'] if isinstance(a, dict))}): every row is "
                    f"reported under the amended protocol and, in validation.json 'version_1', as first fixed")
        if fixed_text["status"] == "changed":
            warn.append("The protocol's fixed text was rewritten, not amended: " + fixed_text["text"])
    # a record-only amendment may name the deramp polygon each run uses (2026-10-08/a-deramp-polygon): a run
    # whose config names another one is not the run the protocol records
    dp_want = ((amendment_switches(P).get("run_deramp_polygon") or {}) if run else {})
    dp_want = dp_want.get(run) if isinstance(dp_want, dict) else None
    if dp_want:
        dp_have = str(getattr(cfg, "deramp_exclude_geojson", "") or "").strip()
        if Path(dp_have).name != Path(str(dp_want)).name:
            warn.append(f"Run {run}'s deramp polygon is recorded as {dp_want} (amendment "
                        f"{amendment_switches(P)['_by'].get('run_deramp_polygon')}), but this config names "
                        f"{dp_have or 'none'}: these are not the protocol's results for {run}")
    tw = (P.get("comparisons") or {}).get("tahmoor_w_units")
    tw_runs = (tw.get("runs") if isinstance(tw, dict) else None) or []
    if (points is None and "tahmoor_w_units" in (P.get("comparisons") or {}) and T1_POINTS.exists()
            and (run is None or not tw_runs or run in tw_runs)):
        warn.append(f"The digitised Tahmoor West units exist ({T1_POINTS}) but none were given (--points, or the "
                    f"config's validation_points), so the tahmoor_w_units row is not computed")

    prod = InsarProduct(cfg)
    ref_rec, ref_warn = product_reference(cfg, prm)
    warn += ref_warn
    floors = [prm["floor"]] + ([f for f in prm["sensitivity_floors"] if f != prm["floor"]] if sensitivity else [])

    # ---- truth
    w0, w1 = prm["window"]
    pad = np.timedelta64(prm["match_days"] + 1, "D")
    window = (str(np.datetime64(w0) - pad), str(np.datetime64(w1) + pad))
    ids = list(prm["stations"])
    ngl, bad = load_ngl(ids, window, cache_dir, prm, gt)
    sources = {"ngl": ngl}
    if prm["design"] == "reference_station" and "ga" in prm.get("source_order", []):
        ga_path = Path(ga_xlsx) if ga_xlsx else None
        try:
            if ga_path is None:
                ga_path = gt.fetch_ga_xlsx(cache_dir)       # "fetched at analysis time" (protocol A)
            ga, bad_ga = load_ga(ga_path, ids, window, prm, gt)
            sources["ga"] = ga
            for k, why in bad_ga.items():
                bad.setdefault(f"{k} (GA)", why)
        except Exception as exc:                          # noqa: BLE001 - the run says so and uses NGL
            warn.append(f"The GA CEMP workbook could not be had ({exc}): the GA-sourced comparisons fall back to "
                        "NGL, which the protocol ranks second")
    from pyproj import Geod
    for src in sources:
        for sid, st in sources[src].items():
            pr_ = prm["stations"][sid]
            if st.get("lat") is not None and st.get("lon") is not None:
                _, _, dist = Geod(ellps="WGS84").inv(pr_["lon"], pr_["lat"], float(st["lon"]), float(st["lat"]))
                st["provenance"]["protocol_to_file_m"] = _f(dist, 1)
                if dist > 100:
                    warn.append(f"{sid}: the truth file's position is {dist:.0f} m from the protocol's")
            if st["provenance"].get("ends_before_window_end"):
                warn.append(f"{sid} ({src}): the record ends {st['provenance'].get('last_epoch_in_file')}, before "
                            f"the window does; its comparisons stop there")

    # ---- InSAR at the stations (and at the CSV truth points, when given)
    sample_pts = {s: prm["stations"][s] for s in ids}
    pts_truth = pts_truth_v1 = None
    truth_rules = None
    if points:
        pts_file = gt.load_points_csv(points)
        sw_now = prm.get("amend") or {}
        # amendment 2026-10-08/p23-1 (digitising sigma) and /p23-4 (Site 13 split at its relocation); the
        # protocol as first fixed (version_1) reads the same file with the first-fixed sigma and no split
        pts_truth, sig_rec = digitised_truth_sigma(points, pts_file, sw_now.get("digitised_sigma"))
        pts_truth, split_rec = split_truth_units(pts_truth, sw_now.get("unit_splits"))
        truth_rules = {"sigma": {**sig_rec, "amendment": (sw_now.get("_by") or {}).get("digitised_sigma")},
                       "unit_splits": {"split": split_rec, "amendment": (sw_now.get("_by") or {}).get("unit_splits")}
                       if sw_now.get("unit_splits") else None}
        if P.get("amendments"):
            pts_truth_v1, sig_v1 = digitised_truth_sigma(points, pts_file, None)
            truth_rules["sigma_as_first_fixed"] = sig_v1
        for pt, _df in pts_truth + (pts_truth_v1 or []):
            sample_pts[pt["id"]] = {"lat": float(pt["lat"]), "lon": float(pt["lon"])}
    samples = sample_points(prod, sample_pts, prm["radius_m"], floors, prm["min_cells"], prm["min_independent_pixels"])
    up_samples = sample_up(up, sample_pts, prm["radius_m"], prm["min_cells"]) if up else None
    # the GNSS daily sigma each station carries, in LOS (its own LOS vector) and Up, and which rule gave it
    # (robust; under amendment 2026-10-08/4 the larger of robust and formal)
    daily_sigma = {}
    for src in sources:
        for sid, st in sources[src].items():
            los = samples.get(sid, {}).get(prm["floor"], {}).get("los") if sid in samples else None
            comp = {"up": np.array([0.0, 0.0, 1.0])}
            if los is not None and np.all(np.isfinite(los)):
                comp["los"] = np.asarray(los, float)
            daily_sigma[sid if src == "ngl" else f"{sid} ({src})"] = {
                c: {k: (_f(v, 3) if isinstance(v, float) else v) for k, v in daily_sigma_record(st, vec, prm).items()}
                for c, vec in comp.items()}

    # ---- C-alias (F2's flag, per comparison interval) and C-events (peak ratios): protocol A's honesty rows
    flag, alias, alias_rec, events_in = (None, "", False), None, None, None
    alias_v1 = None
    if prm["design"] == "reference_station":
        flag = read_flag_layer(cfg, prod.shape)
        summ = read_alias_summary(cfg)
        fv = ((summ or {}).get("harness") or {}).get("flagged_values") or ALIAS_FLAGGED
        # the coherence hole as amended (2026-10-08/p23-2: relative to its ring) or as first fixed
        hole_now = "amended" if (prm.get("amend") or {}).get("coherence_hole") == "relative_to_ring" else "as_first_fixed"
        alias = IntervalFlags(cfg, prod, fv, version=hole_now)
        if P.get("amendments") and hole_now != "as_first_fixed":
            alias_v1 = IntervalFlags(cfg, prod, fv, version="as_first_fixed")
        alias_rec = {"summary": summ, "whole_record_layer": flag[1],
                     "interval_flags": {"read": alias.ok, "what": alias.what, "why_not": alias.why, "pairs": alias.pairs,
                                        "flagged_values": list(alias.flagged), "hole_rule_version": hole_now,
                                        "record_hole_rule": alias.hole_rule}}
        if alias.ok and hole_now == "amended" and not (alias.hole_rule or {}).get("kind", "").startswith("relative"):
            warn.append("The per-pair aliasing record was made with the coherence hole as first fixed (it predates "
                        "amendment 2026-10-08/p23-2): measure it again (aliasing.pre_unwrap) for the amended flag")
        elif alias.ok and hole_now == "amended":
            # the record's own hole rule against the protocol's stated ratio and cell (amendment 2026-10-08/p23-2)
            sw_h, hr_ = prm.get("amend") or {}, alias.hole_rule or {}
            want_r, want_b = sw_h.get("coherence_hole_ratio"), sw_h.get("coherence_hole_block_look_cells")
            off = []
            if want_r is not None and (hr_.get("ratio") is None or abs(float(hr_["ratio"]) - float(want_r)) > 5e-4):
                off.append(f"ratio {hr_.get('ratio')} (the protocol's {want_r})")
            if want_b is not None and f"{int(want_b)} x {int(want_b)}" not in str(hr_.get("cell", "")):
                off.append(f"cell '{hr_.get('cell')}' (the protocol's mean over {int(want_b)} x {int(want_b)} look cells)")
            if off:
                warn.append("The per-pair aliasing record's coherence hole is not the one amendment 2026-10-08/p23-2 "
                            "states: " + "; ".join(off) + ". Measure it again (aliasing.pre_unwrap)")
                alias_rec["interval_flags"]["hole_rule_differs_from_protocol"] = off
        if alias.ok:
            flag = (flag[0], alias.what + (f"; whole-record layer beside it: {flag[1]}" if flag[0] is not None else ""),
                    True)
        elif flag[0] is not None and flag[2]:
            warn.append("The aliasing flag is the whole-record layer (every pair), which flags more than any one "
                        f"comparison interval: the per-pair record was not read ({alias.why})")
        rc_ = (summ or {}).get("record_check") or {}
        if alias.ok and rc_.get("checked") and not rc_.get("complete", True):
            warn.append("The aliasing record does not cover every pair MintPy read (aliasing_risk.json "
                        "'record_check'): their cells may be flagged too little")
        alias_rec["interval_flags"]["stale_pair_files_left_out"] = alias.stale_files
        events_in = event_inputs(P, prm, cfg, prod, outlines=outlines, lines=lines, run=run)
    stable = None
    if prm["design"] == "reference_station" and ((prm.get("amend") or {}).get("event_change_sigma") == "stable_ground"
                                                 or (prm.get("amend") or {}).get("flag_selectivity_max_stable_share")
                                                 is not None):
        avoid = [(float(v["lat"]), float(v["lon"])) for v in sample_pts.values()]
        outl = outlines or (str(getattr(cfg, "event_polygons", "") or "").strip() or None)
        stable = stable_placements(prod, cfg, prm, avoid, outl)
        if stable.get("why"):
            warn.append("No stable-ground interval for the event changes (amendment 2026-10-08/1): " + stable["why"])
        if not outl and T1_OUTLINES.exists():
            # the amendment's own rule: the stable ring once the T1 outlines exist, the reference area until then
            warn.append(f"Amendment 2026-10-08/1 measures the event changes' 95% interval on the stable ring once the "
                        f"T1 panel outlines exist, and they do ({T1_OUTLINES}); none was given (--outlines, or the "
                        f"config's event_polygons), so this run used the reference area instead. Run `insar validate` "
                        f"again with --outlines for the protocol's numbers.")
        if outl:
            stable["outlines_file"] = {"path": str(outl), "sha256": _sha256(Path(outl)) if Path(outl).exists() else None}
    raw, rows, sens, verdict = _compare(P, prm, cfg, prod, sources, samples, floors, pts_truth, flag, up,
                                        up_samples, run, stable, alias, events_in)
    v1 = None
    if P.get("amendments"):
        P1 = version_1(P)
        prm1 = protocol_params(P1, run, getattr(cfg, "backend", None))
        _r1, rows1, _s1, verdict1 = _compare(P1, prm1, cfg, prod, sources, samples, [prm1["floor"]],
                                            pts_truth_v1 if pts_truth_v1 is not None else pts_truth,
                                            flag, up, up_samples, run, None, alias_v1 or alias, events_in)
        v1 = {"what": "every row and the route verdict under the protocol as first fixed on 8 Oct 2026 (its "
                        "amendments set aside), at the primary coherence floor",
              "verdict": verdict1,
              "rows": {n: {k: r.get(k) for k in ("grade", "label", "why", "metrics") if k in r}
                       for n, r in rows1.items() if isinstance(r, dict)}}
    cov = coverage(prod, prm["bbox_wsen"], floors, epoch_coverage)
    sampling = {}
    for pid, per in samples.items():
        if pid.startswith("_"):
            continue
        sampling[pid] = {str(fl): {"cells_within_radius": s["cells_within_radius"], "reliable_cells": s["reliable_cells"],
                                   "dates_with_value": int(np.isfinite(s["series"]).sum()),
                                   "median_cells_per_date": _f(np.median(s["n_cells"]), 1),
                                   "median_independent_per_date": _f(np.median(s["n_independent"]), 1),
                                   "los_vector_enu": [_f(v, 4) for v in s["los"]], "incidence_deg": _f(s["incidence"], 2),
                                   "stated_rate_sigma_mm_yr": _f(s["stated_rate_sigma"], 3)}
                         for fl, s in per.items()}
    for pid, s in sampling.items():
        if s[str(prm["floor"])]["dates_with_value"] == 0:
            warn.append(f"{pid}: no SAR date meets the sampling minimum at the primary floor (no InSAR value)")
    res = {
        "harness": {"module": "insar_pipeline/validation.py", "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "code": _version_label()},
        "protocol": {"id": prm["protocol"], "version": prm["version"], "run": run, "path": P["_path"],
                     "sha256": P["_sha256"], "fixed": fixed, "inherited": P.get("_inherited"), "design": prm["design"],
                     "amendments": P.get("amendments") or [], "fixed_text": fixed_text,
                     "amendment_keys": amendment_switches(P)},
        "parameters": {k: v for k, v in prm.items() if k not in ("hinge_days",)},
        "product": {"config": str(config_path) if config_path else None, "project_dir": str(cfg.project_dir),
                    "backend": getattr(cfg, "backend", None), **prod.describe(),
                    "relative_figures": ref_rec,
                    "pixel_id_method": samples.get("_pixel_id_method")},
        "truth": {"stations": {(s if src == "ngl" else f"{s} ({src})"): st["provenance"]
                               for src in sources for s, st in sources[src].items()},
                  "not_loaded": bad, "cache_dir": str(cache_dir), "daily_sigma": daily_sigma,
                  "points_file": ({"path": str(points), "from": points_from,
                                   "sha256": _sha256(Path(points)) if Path(points).exists() else None}
                                  if points else None),
                  "points_rules": truth_rules},
        "sampling": sampling,
        "coverage": cov,
        "rows": rows,
        "sensitivity": sens,
        "verdict": verdict,
        "power_statement": ((P.get("statistics_reported") or {}).get("power_statement")),
        "warnings": warn,
    }
    if v1 is not None:
        res["version_1"] = v1
    if stable is not None:
        res["stable_ground"] = {k: v for k, v in stable.items() if k not in ("lat", "lon", "series", "los_up")}
    if alias_rec is not None:
        res["aliasing"] = alias_rec
    if events_in is not None:
        xc = events_crosscheck(cfg, prm, samples, raw["by_floor"][prm["floor"]].get("comparisons") or [], prod,
                               sample_pts)
        res["events"] = {"lines_file": events_in.get("lines_file"), "outlines_file": events_in.get("outlines_file"),
                         "why": events_in.get("why"), "values_at_crosscheck": xc}
        if xc.get("n") and not xc.get("agree"):
            warn.append(f"events.values_at and the harness's own sampling differ by up to {xc['max_abs_diff_mm']} mm "
                        "at the compared points: one of the two samples differently")
    if prm["design"] == "network":
        prim = raw["by_floor"][prm["floor"]]
        res["detail"] = {"los": _clean(prim["los"]), "vertical_assumed": _clean(prim.get("vertical_assumed")),
                         "los_vector_variation": prim.get("los_vector_variation")}
    else:
        prim = raw["by_floor"][prm["floor"]]
        res["detail"] = {"reference_choice": prim["reference_choice"], "comparisons": prim["comparisons"],
                         "flag_layer": raw.get("flag_layer")}
    # ---- write
    (out_dir / "validation.json").write_text(json.dumps(_clean(res), indent=1, default=_json_default))
    write_epochs_csv(out_dir / "validation_epochs.csv", raw.get("epoch_table"))
    res["plots"] = write_plots(out_dir, res, raw)
    write_markdown(out_dir / "validation.md", res)
    js = json.loads((out_dir / "validation.json").read_text())
    js["plots"] = res["plots"]
    (out_dir / "validation.json").write_text(json.dumps(js, indent=1, default=_json_default))
    return res


def _version_label():
    try:
        from . import version
        return version.label()
    except Exception:                                         # noqa: BLE001
        return None


def default_out(cfg, P, run) -> Path:
    """<project>/validation/protocol_<id>[_<run>]"""
    return Path(cfg.project_dir) / "validation" / (f"protocol_{P['protocol']}" + (f"_{run}" if run else ""))


def run(cfg) -> Path:
    """The pipeline's experiment stage (pipeline.run_validation, V3): the
    protocol named by cfg.validation_protocol on this run, written under
    default_out. The protocol's run (B1, A3, ...) is found from the config's
    file name (cfg._source_path, set by load_config). Raises when the
    validation cannot be made, so the stage fails rather than reading done."""
    proto = str(getattr(cfg, "validation_protocol", "") or "").strip()
    if not proto:
        raise ValueError("the config names no validation_protocol")
    src = getattr(cfg, "_source_path", None)
    P = load_protocol(proto)
    run_id = _find_run(P, src)
    out = default_out(cfg, P, run_id)
    validate(cfg, proto, out, run=run_id, config_path=src)
    return out


def main(argv=None, cfg=None) -> int:
    ap = argparse.ArgumentParser(
        prog="insar validate",
        description="Compare a finished run's InSAR with independent ground truth under a fixed protocol. "
                    "Reads the run's products and the truth; writes validation.json, validation.md, "
                    "validation_epochs.csv and plots into --out (default: a 'validation' folder inside the "
                    "project). No product or client figure of the run is changed.")
    ap.add_argument("config", help="the run's config (.yaml)")
    ap.add_argument("--protocol", required=True, help="the protocol (.yaml), e.g. review_2026-10-08/protocol_B1.yaml")
    ap.add_argument("--out", help="output folder (default: <project>/validation/protocol_<id>[_<run>])")
    ap.add_argument("--run", help="the protocol's run id (default: found from the config's file name)")
    ap.add_argument("--cache", help="folder for downloaded GNSS (default: <out>/gnss_cache)")
    ap.add_argument("--ga-xlsx", help="Geoscience Australia CEMP xlsx (protocol A)")
    ap.add_argument("--points", help="digitised or survey truth points (CSV, ground_truth.load_points_csv), e.g. "
                                     "the Tahmoor West units. Default: the config's validation_points")
    ap.add_argument("--up", help="D1's decomposed Up series (decompose.py's vertical_timeseries_decomposed.h5; "
                                 "read on its own grid and epochs)")
    ap.add_argument("--outlines", help="digitised panel outlines (GeoJSON, PLAN T1): protocol A's stable ring, where "
                                       "the event changes' stable-ground intervals are measured (amendment 2026-10-08/1), "
                                       "and the Tahmoor West peak (outlines named W1..W4). Default: the config's "
                                       "event_polygons")
    ap.add_argument("--lines", help="digitised end-of-panel survey lines (GeoJSON, PLAN T1), each naming its panel "
                                    "(e.g. 'LW902'; for LW708 also 'ARTC' or 'M31'): protocol A's Appin peak ratio. "
                                    "Default: the config's event_lines")
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the coherence-floor sensitivity runs")
    ap.add_argument("--no-epoch-coverage", action="store_true", help="skip the per-date coverage (reads the whole cube)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if cfg is None:
        from .config import load_config
        cfg = load_config(a.config)
    P = load_protocol(a.protocol)
    run = a.run or _find_run(P, a.config)
    out = Path(a.out) if a.out else default_out(cfg, P, run)
    try:
        res = validate(cfg, a.protocol, out, run=run, cache_dir=a.cache, ga_xlsx=a.ga_xlsx, points=a.points,
                       up=a.up, sensitivity=not a.no_sensitivity, epoch_coverage=not a.no_epoch_coverage,
                       config_path=a.config, outlines=a.outlines, lines=a.lines)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Validation not made: {exc}")
        return 1
    v = res["verdict"]
    print(f"Protocol {res['protocol']['id']}" + (f", run {run}" if run else "") + f": {res['protocol']['fixed']['text']}")
    for name, g in v["rows"].items():
        print(f"  {name:26s} {GRADE_TEXT.get(g, g)}")
    print(f"Route: {GRADE_TEXT.get(v['route']['grade'], v['route']['grade'])} - {v['route']['text']}")
    print(f"Written to {out}: validation.json, validation.md, validation_epochs.csv"
          + (", " + ", ".join(res.get("plots") or []) if res.get("plots") else ""))
    return 0
