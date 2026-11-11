"""evidence.yaml: what the program has been shown to measure, and the readiness
tier an output earns from it (review_2026-11-07/PLAN.md, "Readiness tiers and
the rule that sets them"; S0.4, 8 Nov 2026).

load() reads and checks evidence.yaml (the only place verdicts live);
output_tier() applies the plan's seven rules to one domains.Output. Nothing in
the report path calls it yet: the default study is grandfathered (D5), so its
computed tiers are recorded for provenance and LIMITATIONS later, never
enforced (but the longwall refusal, Phase 1c). tests/test_domains.py recomputes
them against the expected-tier table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import domains

PKG = Path(__file__).resolve().parent
EVIDENCE_YAML = PKG / "evidence.yaml"

VERDICTS = ("PASS", "FAIL", "MARGINAL", "UNTESTABLE", "PENDING")
QUANTITY_VERDICTS = ("PASS", "FAIL", "MARGINAL", "HELD", "REPORTED", "UNTESTABLE", "PENDING")
TIERS = ("VALIDATED", "INDICATIVE", "SCREENING", "RESEARCH", "REFUSED")
GEOMETRIES = ("single", "asc+desc")
# a route's cap (rule 2), from config.ROUTE_TAG: the official route is not capped
ROUTE_TAG_CAP = {"official": None, "screening": "SCREENING", "comparison": "RESEARCH"}

# rule 1: a quantity's grade -> the most it can give; the lowest grade wins
_GRADE_TIER = {"PASS": "VALIDATED", "MARGINAL": "INDICATIVE", "HELD": "INDICATIVE", "REPORTED": "INDICATIVE",
               "UNTESTABLE": "RESEARCH", "PENDING": "RESEARCH", "FAIL": "REFUSED"}
# order for "lowest": REFUSED is below everything (a cap never lifts it)
_RANK = {"REFUSED": 0, "RESEARCH": 1, "SCREENING": 2, "INDICATIVE": 3, "VALIDATED": 4}

_RECORD_KEYS = {"title", "protocol", "protocol_sha256", "local", "results", "runs", "verdict", "quantities", "note",
                "sign", "max_rate_mm_yr", "rate_basis", "rate_source", "regime", "ground", "footprint", "band",
                "route", "geometry", "independent_truth"}
_REQUIRED = {"title", "protocol", "verdict", "quantities", "sign", "max_rate_mm_yr", "rate_basis", "rate_source",
             "ground", "footprint", "band", "route", "geometry", "independent_truth"}


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class Record:
    id: str
    title: str
    protocol: str
    verdict: str
    quantities: dict
    sign: str
    max_rate_mm_yr: float | None
    rate_basis: str
    ground: tuple
    footprint: str
    band: str
    route: str
    geometry: str
    independent_truth: bool
    regime: str = ""
    local: bool = False
    protocol_sha256: str = ""


@dataclass(frozen=True)
class Evidence:
    records: dict
    overrides: dict = field(default_factory=dict)
    pending_overrides: dict = field(default_factory=dict)
    transfers: dict = field(default_factory=dict)


def _check_record(rid: str, r: dict) -> Record:
    where = f"evidence.yaml {rid}"
    if not isinstance(r, dict):
        raise EvidenceError(f"{where}: not a mapping")
    extra, missing = set(r) - _RECORD_KEYS, _REQUIRED - set(r)
    if extra or missing:
        raise EvidenceError(f"{where}: unknown keys {sorted(extra)}, missing {sorted(missing)}")
    if r["verdict"] not in VERDICTS:
        raise EvidenceError(f"{where}: verdict {r['verdict']!r} not one of {VERDICTS}")
    q = r["quantities"]
    if not isinstance(q, dict) or not q:
        raise EvidenceError(f"{where}: quantities must be a non-empty mapping")
    bad = {k: v for k, v in q.items() if v not in QUANTITY_VERDICTS}
    if bad:
        raise EvidenceError(f"{where}: quantity grades {bad} not one of {QUANTITY_VERDICTS}")
    if r["sign"] not in domains.SIGNS:
        raise EvidenceError(f"{where}: sign {r['sign']!r} not one of {domains.SIGNS}")
    if r["footprint"] not in domains.FOOTPRINTS:
        raise EvidenceError(f"{where}: footprint {r['footprint']!r} not one of {domains.FOOTPRINTS}")
    if r["rate_basis"] not in domains.COMPONENTS:
        raise EvidenceError(f"{where}: rate_basis {r['rate_basis']!r} not one of {domains.COMPONENTS}")
    if r["geometry"] not in GEOMETRIES:
        raise EvidenceError(f"{where}: geometry {r['geometry']!r} not one of {GEOMETRIES}")
    if not isinstance(r["independent_truth"], bool):
        raise EvidenceError(f"{where}: independent_truth must be true or false")
    m = r["max_rate_mm_yr"]
    if m is not None and (isinstance(m, bool) or not isinstance(m, (int, float)) or m <= 0):
        raise EvidenceError(f"{where}: max_rate_mm_yr must be a positive number or null")
    reg = r.get("regime", "") or ""
    if reg and reg not in domains.FAST_REGIMES:
        raise EvidenceError(f"{where}: regime {reg!r} not one of {domains.FAST_REGIMES}")
    local = bool(r.get("local", False))
    sha = r.get("protocol_sha256", "") or ""
    if not local and not sha:
        raise EvidenceError(f"{where}: a tracked protocol needs protocol_sha256 (or local: true)")
    ground = r["ground"]
    if not isinstance(ground, list) or not ground or not all(isinstance(g, str) for g in ground):
        raise EvidenceError(f"{where}: ground must be a non-empty list of class names")
    return Record(id=rid, title=r["title"], protocol=r["protocol"], verdict=r["verdict"], quantities=dict(q),
                  sign=r["sign"], max_rate_mm_yr=None if m is None else float(m), rate_basis=r["rate_basis"],
                  ground=tuple(ground), footprint=r["footprint"], band=str(r["band"]), route=str(r["route"]),
                  geometry=r["geometry"], independent_truth=r["independent_truth"], regime=reg, local=local,
                  protocol_sha256=sha)


def _check_override(where: str, ov: dict, ids) -> dict:
    out = {}
    for k, v in (ov or {}).items():
        if k not in ids:
            raise EvidenceError(f"evidence.yaml {where}: {k!r} is not an evidence id")
        if not isinstance(v, dict) or v.get("tier") not in TIERS or not v.get("why"):
            raise EvidenceError(f"evidence.yaml {where} {k}: needs tier (one of {TIERS}) and why")
        if where == "overrides" and not v.get("signed"):
            raise EvidenceError(f"evidence.yaml overrides {k}: an applied override must be signed")
        out[k] = dict(v)
    return out


def parse(doc: dict) -> Evidence:
    if not isinstance(doc, dict) or doc.get("schema") != 1:
        raise EvidenceError("evidence.yaml: expected schema: 1")
    extra = set(doc) - {"schema", "records", "overrides", "pending_overrides", "transfers"}
    if extra:
        raise EvidenceError(f"evidence.yaml: unknown top-level keys {sorted(extra)}")
    recs = {str(k): _check_record(str(k), v) for k, v in (doc.get("records") or {}).items()}
    if not recs:
        raise EvidenceError("evidence.yaml: no records")
    transfers = {}
    for study, t in (doc.get("transfers") or {}).items():
        if not isinstance(t, dict) or not t.get("signed") or not t.get("quantities") or not t.get("evidence"):
            raise EvidenceError(f"evidence.yaml transfers {study}: needs quantities, evidence and signed")
        unknown = [e for e in t["evidence"] if e not in recs]
        if unknown:
            raise EvidenceError(f"evidence.yaml transfers {study}: unknown evidence {unknown}")
        transfers[str(study)] = {"quantities": tuple(t["quantities"]), "evidence": tuple(t["evidence"]),
                                 "signed": str(t["signed"])}
    return Evidence(records=recs, overrides=_check_override("overrides", doc.get("overrides"), recs),
                    pending_overrides=_check_override("pending_overrides", doc.get("pending_overrides"), recs),
                    transfers=transfers)


def load(path: str | Path | None = None) -> Evidence:
    """Read and check evidence.yaml (a key given twice is an error, as in configs)."""
    import yaml

    from .config import _UniqueKeyLoader
    p = Path(path) if path else EVIDENCE_YAML
    try:
        doc = yaml.load(p.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError, ValueError) as e:
        raise EvidenceError(f"{p}: {e}") from e
    return parse(doc)


def _cap(tier: str, most: str | None) -> str:
    if most is None or tier == "REFUSED":
        return tier
    return most if _RANK[most] < _RANK[tier] else tier


def _sign_covered(rec: Record, out_sign: str) -> bool:
    """The evidence covers every sign the output reports."""
    return rec.sign == "any" or rec.sign == out_sign


def _sign_overlaps(rec: Record, out_sign: str) -> bool:
    """The output reports some sign the evidence was graded on (so a FAIL there applies)."""
    return rec.sign == "any" or out_sign == "any" or rec.sign == out_sign


def output_tier(study: domains.Study, output: domains.Output, ev: Evidence, *, route: str = "miaplpy",
                tracks: int = 1, regime: str = "", near_known_movement: bool = False,
                measured_rate_mm_yr: float | None = None, expected_max_rate_mm_yr: float = 0.0,
                ground: str | None = None, measured_vertical_mm_yr: float | None = None) -> tuple[str, list[str]]:
    """(tier, reasons) for one output under the plan's rules 1-7.

    route: the config backend; tracks: 1 or 2 viewing directions; regime: a fast
    regime the output sits in ("" = none); near_known_movement: inside, or within
    one equivalent radius of, a known_movement or source polygon;
    measured_rate_mm_yr: the measured rate in the output's own component,
    compared only with ceilings in that component; measured_vertical_mm_yr: its
    vertical equivalent (LOS / cos incidence), compared with vertical ceilings
    whatever the output's component; expected_max_rate_mm_yr: the config's
    expected vertical rate, compared only with vertical ceilings; ground: the
    project's ground class (rule 4, new studies only; None = not known, which
    caps a new study at SCREENING).

    A signed override (rule 7) replaces its entry's rule-1 grade; the caps of
    rules 2-6 still apply to it."""
    from .config import ROUTE_TAG
    missing = [e for e in output.evidence if e not in ev.records]
    if missing:
        raise EvidenceError(f"output {output.key}: unknown evidence ids {missing}")
    # rule 3, first: a fast regime refuses whatever was measured, with or without evidence
    if regime and regime in domains.FAST_REGIMES:
        return "REFUSED", [f"fast regime {regime}: C-band reads it low"]
    why = []
    # rule 1: the lowest grade for the output's quantity among its evidence ids
    used = [ev.records[e] for e in output.evidence if output.quantity in ev.records[e].quantities]
    if not used:
        return "RESEARCH", [f"no evidence grades {output.quantity!r}"]
    # rule 4 (new studies): grade on the entries whose ground matches the project's;
    # with none, all of them, capped below
    graded = used                                            # the envelope (rule 3) is every entry's
    on_ground = [rec for rec in used if ground is not None and ground in rec.ground]
    if not study.grandfathered and on_ground:
        used = on_ground
    tier = "VALIDATED"
    for rec in used:
        ov = ev.overrides.get(rec.id)
        if ov:                                               # rule 7: signed overrides only
            tier = min(tier, ov["tier"], key=_RANK.get)
            why.append(f"override {rec.id}: {ov['tier']} ({ov['signed']})")
            continue
        grade = rec.quantities[output.quantity]
        t = _GRADE_TIER[grade]
        # rule 3 in part: a FAIL refuses only inside the regime it failed in
        in_regime = (_sign_overlaps(rec, output.sign) and (not rec.regime or rec.regime == regime)
                     and domains.FOOTPRINTS.index(output.footprint) >= domains.FOOTPRINTS.index(rec.footprint))
        if grade == "FAIL" and not in_regime:
            t = "SCREENING"
        if not rec.independent_truth:
            t = _cap(t, "SCREENING")
            why.append(f"{rec.id} truth is not independent")
        tier = min(tier, t, key=_RANK.get)
        why.append(f"{rec.id} {output.quantity} {grade}")
    # rule 2: the route
    tag = ROUTE_TAG.get(str(route).lower(), "comparison")
    most = ROUTE_TAG_CAP[tag]
    if most:
        tier = _cap(tier, most)
        why.append(f"route {route} ({tag})")
    # rule 3: the regime envelope, each ceiling in its own component, over every
    # entry the output rests on (not only those on the project's ground)
    for rec in graded:
        if not _sign_covered(rec, output.sign):
            tier = _cap(tier, "SCREENING")
            why.append(f"{rec.id} covers sign {rec.sign}, not {output.sign}")
        if domains.FOOTPRINTS.index(output.footprint) < domains.FOOTPRINTS.index(rec.footprint):
            tier = _cap(tier, "SCREENING")
            why.append(f"{rec.id} footprint {rec.footprint} is larger than {output.footprint}")
        if rec.max_rate_mm_yr is not None:
            if expected_max_rate_mm_yr and rec.rate_basis == "vertical" and expected_max_rate_mm_yr > rec.max_rate_mm_yr:
                return "REFUSED", why + [f"expected rate {expected_max_rate_mm_yr} above {rec.id}'s ceiling"]
            if (measured_rate_mm_yr is not None and rec.rate_basis == output.component
                    and abs(measured_rate_mm_yr) > rec.max_rate_mm_yr):
                tier = _cap(tier, "SCREENING")
                why.append(f"measured rate above {rec.id}'s {rec.max_rate_mm_yr} mm/yr ({rec.rate_basis})")
            if (measured_vertical_mm_yr is not None and rec.rate_basis == "vertical"
                    and abs(measured_vertical_mm_yr) > rec.max_rate_mm_yr):
                tier = _cap(tier, "SCREENING")
                why.append(f"vertical rate above {rec.id}'s {rec.max_rate_mm_yr} mm/yr")
    # rule 4: ground (new studies only); an unknown ground is not on the evidence
    if not study.grandfathered and (ground is None or not any(ground in rec.ground for rec in used)):
        tier = _cap(tier, "SCREENING")
        why.append(f"ground {ground} not on the evidence" if ground else "ground class not known")
    # rule 5: geometry
    if "both_directions" in output.needs and tracks < 2:
        tier = _cap(tier, "INDICATIVE")
        why.append("needs both directions")
    if output.component == "vertical" and tracks < 2 and near_known_movement:
        tier = _cap(tier, "INDICATIVE")
        why.append("single-track vertical near known movement")
    # a study not offered in the builder (missing producers, waiting on a gate) is research only
    if study.builder_status != "offered":
        tier = _cap(tier, "RESEARCH")
        why.append(f"study {study.key} not offered: {study.builder_reason}")
    # rule 6: a new study needs a signed transfer
    if not study.grandfathered:
        t = ev.transfers.get(study.key)
        if not (t and output.quantity in t["quantities"] and set(output.evidence) <= set(t["evidence"])):
            tier = _cap(tier, "SCREENING")
            why.append("no signed transfer")
    return tier, why


def card(study: domains.Study, ev: Evidence | None = None, *, tracks: int | None = None,
         ground: str | None = None, route: str = "miaplpy") -> list[str]:
    """What a study will give and will not give, with each output's computed
    tier: the step-6 card, the preflight block and `insar studies`.
    tracks: the project's viewing directions (None = no project: tiers for one)."""
    ev = ev or load()
    project_tracks = tracks
    tracks = tracks or 1
    lines = [f"{study.label} ({study.key}): {study.question}"]
    if study.builder_status != "offered":
        lines.append(f"  Not offered yet: {study.builder_reason}")
    lines.append("  Will give:")
    for o in study.outputs:
        tier, _why = output_tier(study, o, ev, tracks=tracks, ground=ground, route=route)
        if ground is None and not study.grandfathered:
            # a new study's tier depends on the ground class (rule 4): show each one
            # the evidence covers that does better than the unknown-ground tier
            grounds = sorted({g for e in o.evidence if e in ev.records and o.quantity in ev.records[e].quantities
                              for g in ev.records[e].ground})
            per = {g: output_tier(study, o, ev, tracks=tracks, ground=g, route=route)[0] for g in grounds}
            better = {g: t for g, t in per.items() if _RANK[t] > _RANK[tier]}
            if better:
                tier = ("; ".join(f"{t} on {g.replace('_', ' ')} ground" for g, t in better.items())
                        + f"; {tier} elsewhere")
        extra = "" if o.automatic else f" (only when asked: {o.how})"
        need = f" [needs {', '.join(o.needs)}]" if o.needs else ""
        lines.append(f"    - {o.label}: {tier}{need}{extra}")
    if study.cannot:
        lines.append("  Cannot tell you:")
        lines += [f"    - {t} ({tag})" for t, tag in study.cannot]
    if study.not_yet:
        lines.append("  Not yet:")
        lines += [f"    - {t} ({tag})" for t, tag in study.not_yet]
    if study.directions != "one":
        lines.append("  Both viewing directions " + ("required" if study.directions == "both_required"
                                                     else "recommended")
                     + (f" (this project: {project_tracks})" if project_tracks else ""))
    lines += [f"  {t}" for t in study.fixed_sentences]
    if study.layer_kinds:
        lines.append("  Layers: " + "; ".join(f"{L.label} ({L.config_field}"
                                              + (f", kind {L.key}" if L.config_field == "feature_layers" else "")
                                              + ")" for L in study.layer_kinds))
    if study.presets:
        lines.append("  Set at creation: " + ", ".join(f"{k}: {v}" for k, v in study.presets))
    return lines

