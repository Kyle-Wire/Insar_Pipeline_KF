"""The study registry (domains.py), evidence.yaml and the tier rule (evidence.py).

S0.4 of review_2026-11-07/PLAN.md, 8 Nov 2026, Gate 0(c). Guards:
  - the registry's shape: known words only, presets only from PRESET_KEYS
    (never reference_mode or control_ground_label), real config fields, and a
    module free of I/O and of the rest of the package;
  - the default study IS today's wording: the old names in report_insights,
    client_report and areas read it (the golden builds check every byte);
  - evidence.yaml: it loads, its tracked protocols still hash to the recorded
    sha256, and every id an output names exists;
  - the id tags in VALIDATION.md and LIMITATIONS.md: each id tagged at least
    once and at most once per document, and the tagged entry uses the verdict
    word evidence.yaml records;
  - every computed tier against the expected-tier table below, which comes
    from the plan (D5, the default study's section, the rules);
  - names: the hashed dossier names (tests/data/denied_names.sha256) and, when
    the local marker file exists, the client markers are absent from the
    files this registry adds or tags.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import sys
import unicodedata
from dataclasses import replace
from pathlib import Path

import pytest

from insar_pipeline import areas, client_report, config, domains, evidence, report_insights as ri

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client_files import need  # noqa: E402

EV = evidence.load()
D = domains.DEFAULT


# ---------------------------------------------------------------- the registry
def test_the_default_study_is_registered_and_is_the_config_default():
    assert domains.DEFAULT_STUDY == "ground_subsidence" == D.key
    assert domains.study(None) is D and domains.study("") is D and domains.study("Ground_Subsidence") is D
    assert config.PipelineConfig.__dataclass_fields__["study"].default == domains.DEFAULT_STUDY
    assert config.PipelineConfig.__dataclass_fields__["study_registry_version"].default == 0
    assert config._ENUMS["study"] == domains.STUDY_KEYS
    assert len(set(domains.STUDY_KEYS)) == len(domains.STUDIES)
    with pytest.raises(KeyError):
        domains.study("no_such_study")


def test_every_registry_word_is_a_known_one():
    fields = set(config.PipelineConfig.__dataclass_fields__)
    for s in domains.STUDIES:
        assert s.builder_status in domains.BUILDER_STATUS and s.directions in domains.DIRECTIONS
        assert len({o.key for o in s.outputs}) == len(s.outputs), s.key
        for o in s.outputs:
            assert o.kind in domains.OUTPUT_KINDS and o.sign in domains.SIGNS, o.key
            assert o.footprint in domains.FOOTPRINTS and o.component in domains.COMPONENTS, o.key
            assert all(e in EV.records for e in o.evidence), (o.key, o.evidence)
            assert all(t in evidence.TIERS for t, _ in o.client_words), o.key
        for L in s.layer_kinds:
            assert L.role in domains.LAYER_ROLES and L.product in domains.LAYER_PRODUCTS, L.key
            assert L.regime in ("",) + domains.FAST_REGIMES, L.key
            # feature_layers (kinds with no existing field) arrives with its first consumer, Phase 1a
            assert L.config_field in fields, L.config_field
        assert all(tag in domains.CANNOT_TAGS for _, tag in s.cannot)
        assert all(tag in domains.NOT_YET_TAGS for _, tag in s.not_yet)


def test_each_study_writes_a_layer_field_once_and_names_real_producers():
    import importlib
    for s in domains.STUDIES:
        own = [L.config_field for L in s.layer_kinds if L.config_field != "feature_layers"]
        assert len(own) == len(set(own)), f"{s.key}: a config field holds two layer kinds: {own}"
        keys = [L.key for L in s.layer_kinds]
        assert len(keys) == len(set(keys)), s.key
        for o in s.outputs:
            mod, _, fn = o.producer.rpartition(".")
            assert callable(getattr(importlib.import_module(f"insar_pipeline.{mod}"), fn, None)), (s.key, o.producer)
            assert o.automatic or o.how, f"{s.key} {o.key}: say what turns it on"
        if s.builder_status != "offered":
            assert s.builder_reason, s.key
        assert all(k != "site_profile" or v in config.SITE_PROFILES for k, v in s.presets), s.key
        if s.qa_profile:
            assert dict(s.presets).get("site_profile") == s.qa_profile, s.key


def test_presets_only_name_the_four_preset_keys():
    fields = set(config.PipelineConfig.__dataclass_fields__)
    assert set(domains.PRESET_KEYS) <= fields           # each preset is a real setting
    assert not set(domains.NEVER_PRESET) & set(domains.PRESET_KEYS)
    for s in domains.STUDIES:
        keys = [k for k, _ in s.presets]
        assert set(keys) <= set(domains.PRESET_KEYS), (s.key, keys)
        assert len(keys) == len(set(keys)), s.key
    # Gate 0(b): write_project adds only `study` and `study_registry_version`, so
    # the default study can preset nothing
    assert D.presets == ()


def test_domains_is_pure():
    """Standard library only, no I/O, and nothing from the package (areas,
    config and report_insights import it; it must import none of them)."""
    tree = ast.parse((PKG / "domains.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module in ("__future__", "dataclasses"), ast.dump(node)
        elif isinstance(node, ast.Import):
            raise AssertionError(f"domains.py imports {[a.name for a in node.names]}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("open", "exec", "eval", "__import__"), node.func.id
    src = (PKG / "domains.py").read_text()
    code = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    # no grade words and no measured figures: those live in evidence.yaml
    for c in code:
        assert not re.search(r"\b(PASS|FAIL|MARGINAL|HELD|REPORTED|UNTESTABLE|PENDING|VALIDATED|INDICATIVE)\b", c.value), c.value
    for s in domains.STUDIES:
        for text, _ in s.cannot + s.not_yet:
            bare = re.sub(r"evidence [A-Za-z0-9_.]+|\bGate \d+[a-z]?\b|\bV\d\b|\bCO2\b", "", text)
            assert not re.search(r"\d", bare), f"{s.key}: a figure in {text!r} belongs in evidence.yaml"
    assert "verdict" not in src.lower().replace("no verdicts", "")


# ------------------------------------------------ the default study is today's
def test_the_old_names_read_the_default_study():
    w = D.wording
    assert ri.KIND_LABEL == dict(w.feature_noun) and list(ri.KIND_LABEL) == ["underground", "surface", "other", "mapped"]
    assert ri.NEAREST_LABEL == dict(w.nearest_noun)
    assert (ri.NEAR_MINE_KM, ri.ON_LAYER_SHARE, ri.MINING_HORIZONTAL_SHARE) == (0.5, 0.3, 0.35)
    assert ri.NEAR_MINE_KM == D.limits.near_km and ri.MINING_HORIZONTAL_SHARE == D.limits.horizontal_share
    assert dict(ri._CORRECTION_WORDS)["ramp"] == w.correction_ramp
    assert [t for t, _ in ri._CORRECTION_WORDS] == ["ERA5", "HRRR", "GACOS", "tropHgt", "ramp", "demErr", "SET",
                                                    "GIM", "ion"]
    assert areas.MINE_KIND_LABEL == dict(w.source_label)
    assert list(areas.MINE_KIND_LABEL) == ["underground", "surface", "other", "dated"]
    # the report's layer credits map back to the layer keys (report_insights._SOURCE_KEY)
    for key, label in areas.MINE_KIND_LABEL.items():
        assert ri._SOURCE_KEY[label.lower()] == key
    gl = dict(client_report.glossary({"comparison_ground": {"label": "farmland"}}))
    assert gl["Comparison ground"] == w.comparison_ground.format(ground=ri.ground_word({"comparison_ground": {"label": "farmland"}}))


def test_aliases_are_the_same_functions():
    assert ri.study_layers is ri.mine_layers
    assert ri.relation_to_layers is ri.relation_to_mines


@pytest.mark.parametrize("key,sign,shape,start", [
    ("surface", 1, None, "Rising ground on a surface mine"),
    ("underground", 1, None, "Rising ground is usually spoil"),
    ("surface", -1, "slowing", "Backfilled spoil"),
    ("underground", -1, None, "Subsidence over underground workings"),
    ("near_surface", -1, None, "It is beside mapped mining"),
    ("other", -1, None, "It lies on ground already known to move."),
    ("mapped", -1, None, "The mapped mines do not explain it."),
    ("none", 0, None, "The mapped mines do not explain it."),
])
def test_cause_rules_give_todays_sentences(key, sign, shape, start):
    assert ri._cause_base(key, sign, shape).startswith(start)
    if key == "surface" and sign < 0:
        assert "slowing over time, which matches this record. If" in ri._cause_base(key, sign, "slowing")
        assert "slowing over time. If" in ri._cause_base(key, sign, "steady")


# ------------------------------------------------------------------ evidence.yaml
def test_evidence_yaml_holds_the_seed_ids_and_tracked_protocols_still_match():
    assert set(EV.records) == {"B2", "B1_B3", "D1", "A", "B5a", "P0.3b", "LC", "N3", "R1a", "R1a2"}
    for rec in EV.records.values():
        if rec.local:
            # 11 Nov 2026: a local record's protocol is checked where it exists (the
            # machine that ran it), so a frozen local protocol cannot change unseen
            lp = PKG / rec.protocol
            if rec.protocol_sha256 and lp.is_file():
                assert hashlib.sha256(lp.read_bytes()).hexdigest() == rec.protocol_sha256, f"{rec.id}: {rec.protocol} changed"
            continue
        p = need(PKG / rec.protocol, "review records are not shipped in a release")
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec.protocol_sha256, f"{rec.id}: {rec.protocol} changed"


def test_evidence_yaml_rejects_unknown_words(tmp_path):
    import yaml
    doc = yaml.safe_load((PKG / "evidence.yaml").read_text())
    bad = dict(doc, records=dict(doc["records"], B2=dict(doc["records"]["B2"], verdict="OK")))
    with pytest.raises(evidence.EvidenceError, match="verdict"):
        evidence.parse(bad)
    bad = dict(doc, overrides={"LC": {"tier": "SCREENING", "why": "x"}})
    with pytest.raises(evidence.EvidenceError, match="signed"):
        evidence.parse(bad)
    p = tmp_path / "e.yaml"
    p.write_text("schema: 1\nschema: 1\n")
    with pytest.raises(evidence.EvidenceError, match="twice"):
        evidence.load(p)


def test_the_signed_transfer_and_the_unsigned_overrides():
    t = EV.transfers["basin_subsidence"]
    assert t["quantities"] == ("multi_year_rate",) and t["evidence"] == ("B2", "B1_B3") and t["signed"]
    assert EV.overrides == {}                           # nothing signed: the plan's two proposals are pending
    assert set(EV.pending_overrides) == {"P0.3b", "LC"}


# ------------------------------------------------------------------- the id tags
TAG = re.compile(r"\[([A-Za-z0-9_.]+)\](?!\()")            # not a markdown link [x](y)
VERDICT_WORDS = {"PASS": ("pass", "passed", "passes"), "FAIL": ("fail", "failed", "fails", "not passed"),
                 "MARGINAL": ("marginal",), "UNTESTABLE": ("untestable", "not testable", "too few to judge",
                                                           "too small a sample to judge"),
                 "PENDING": ("pending", "not computed", "never run", "not yet run")}
DOCS = ("VALIDATION.md", "LIMITATIONS.md")
# where each id is tagged, and a word its entry must contain (so a tag moved to
# another entry, or dropped from one document, fails)
TAGGED_IN = {"VALIDATION.md": {"B2", "B1_B3", "D1", "A", "B5a"},
             "LIMITATIONS.md": {"A", "B5a", "B1_B3", "LC", "P0.3b", "N3", "R1a", "R1a2"}}
ANCHOR = {"B2": "houston", "B1_B3": "corcoran", "D1": "vertical and east-west", "A": "longwall", "B5a": "nisar l-band",
          "LC": "local change layer", "P0.3b": "east-west", "N3": "phase-linked nisar",
          "R1a": "seasonal", "R1a2": "re-test"}


def _tagged_entries(doc: str) -> list[tuple[str, str]]:
    """(id, entry text) for each tag: a table row, or the bullet line with its
    continuation lines."""
    lines = (PKG / doc).read_text().splitlines()
    out = []
    for i, line in enumerate(lines):
        for m in TAG.finditer(line):
            if m.group(1) not in EV.records:
                continue
            entry = [line]
            if not line.lstrip().startswith("|"):
                start = i                                   # back to the bullet's first line
                while start > 0 and not lines[start].lstrip().startswith("- ") and lines[start].startswith(" "):
                    start -= 1
                entry = lines[start:i + 1]
                for nxt in lines[i + 1:]:
                    if not nxt.strip() or nxt.lstrip().startswith(("- ", "#", "|")) or not nxt.startswith(" "):
                        break
                    entry.append(nxt)
            out.append((m.group(1), " ".join(entry)))
    return out


def test_every_evidence_id_is_tagged_once_with_its_verdict_word():
    seen = {}
    for doc in DOCS:
        tags = _tagged_entries(doc)
        ids = [i for i, _ in tags]
        twice = sorted({i for i in ids if ids.count(i) > 1})
        assert not twice, f"{doc}: tagged more than once: {twice}"
        assert set(ids) == TAGGED_IN[doc], f"{doc}: tagged {sorted(ids)}, expected {sorted(TAGGED_IN[doc])}"
        for rid, text in tags:
            seen.setdefault(rid, []).append(doc)
            assert ANCHOR[rid] in text.lower(), f"{doc} [{rid}] is on an entry about something else: {text[:120]}"
            words = VERDICT_WORDS[EV.records[rid].verdict]
            low = text.lower()
            assert any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in words), \
                f"{doc} [{rid}]: the entry does not say {EV.records[rid].verdict} ({words}): {text[:160]}"
    assert set(seen) == set(EV.records), f"never tagged: {sorted(set(EV.records) - set(seen))}"


def test_no_tag_names_an_unknown_id():
    for doc in DOCS:
        for m in TAG.finditer((PKG / doc).read_text()):
            tag = m.group(1)
            if re.fullmatch(r"[A-Z][A-Za-z0-9_.]*", tag):
                assert tag in EV.records, f"{doc}: [{tag}] is not an evidence id"


# ------------------------------------------------------------- computed tiers
# (output, conditions, expected tier, where the plan says so)
EXPECTED_TIERS = [
    ("multi_year_rate", {}, "VALIDATED", "default study: multi-year sinking rates over report areas (B2 PASS)"),
    # vertical (LOS / cos incidence) was reported, never graded: B2 and B1_B3 vertical_rate REPORTED
    ("multi_year_rate_vertical", {}, "INDICATIVE", "rule 1: vertical rates REPORTED"),
    ("multi_year_rate_vertical", {"near_known_movement": True}, "INDICATIVE", "rule 5 as well"),
    ("multi_year_rate_vertical", {"measured_rate_mm_yr": 30.0}, "SCREENING", "rule 3: above B2's vertical 20 mm/yr"),
    ("multi_year_total", {}, "INDICATIVE", "D5: multi-year totals INDICATIVE"),
    ("series", {}, "INDICATIVE", "D5: series INDICATIVE"),
    ("single_year_rate", {}, "REFUSED", "D5: single-year rates REFUSED"),
    ("feature_search", {}, "SCREENING", "D5: feature search SCREENING"),
    ("rising_ground", {}, "SCREENING", "D5: rising wording SCREENING"),
    # the plan lists trend-shape words as screening, but no evidence entry grades
    # them: rule 1 gives RESEARCH until one does (grandfathered, nothing changes)
    ("trend_shape", {}, "RESEARCH", "rule 1: no evidence"),
    # East-West: indicative only once the pending override is signed
    ("east_west", {"tracks": 2}, "SCREENING", "rule 1: synthetic truth"),
    ("east_west", {"tracks": 1}, "SCREENING", "rule 5 as well"),
    # LIMITATIONS calls it a screening aid; that is the pending LC override, so until it
    # is signed the failed protocol refuses it
    ("local_change", {}, "REFUSED", "rule 3: LC FAIL in its own regime"),
    ("multi_year_rate", {"route": "disp"}, "SCREENING", "rule 2"),
    ("multi_year_rate", {"route": "nisar"}, "RESEARCH", "rule 2"),
    ("multi_year_rate", {"route": "stack"}, "RESEARCH", "rule 2"),
    ("multi_year_rate", {"route": "local"}, "RESEARCH", "rule 2"),
    ("multi_year_rate", {"route": "hyp3"}, "RESEARCH", "rule 2"),
    ("multi_year_rate", {"regime": "active_longwall"}, "REFUSED", "rule 3: a fast regime refuses (A FAIL)"),
    ("trend_shape", {"regime": "active_longwall"}, "REFUSED", "rule 3: even with no evidence"),
    ("multi_year_rate", {"expected_max_rate_mm_yr": 400.0}, "REFUSED", "rule 3: expected rate above the ceiling"),
    # a LOS rate is not compared with B2's vertical ceiling (rule 3: the envelope's own component)
    ("multi_year_rate", {"measured_rate_mm_yr": 30.0}, "VALIDATED", "rule 3: different component"),
]


@pytest.mark.parametrize("key,kw,tier,why", EXPECTED_TIERS, ids=[f"{k}-{t}-{i}" for i, (k, _, t, _) in enumerate(EXPECTED_TIERS)])
def test_computed_tiers_match_the_expected_table(key, kw, tier, why):
    got, reasons = evidence.output_tier(D, D.output(key), EV, **kw)
    assert got == tier, f"{key} {kw}: {got} ({reasons}); expected {tier}: {why}"


# the new studies (PLAN.md catalogue, sections 2-15), one track, phase-linked route:
# (study, output, conditions, tier, why)
NEW_STUDY_TIERS = [
    # groundwater: the signed transfer (D6) covers multi-year LOS rates; graded on the
    # evidence whose ground matches (rule 4): B2 suburban, B1_B3 irrigated farmland (HELD)
    ("basin_subsidence", "multi_year_rate", {"ground": "suburban"}, "VALIDATED", "B2 PASS, signed transfer"),
    ("basin_subsidence", "multi_year_rate", {"ground": "irrigated_farmland"}, "INDICATIVE", "B1_B3 HELD"),
    ("basin_subsidence", "multi_year_rate", {"ground": "rain_fed_cropland"}, "SCREENING", "rule 4: no evidence there"),
    ("basin_subsidence", "multi_year_rate", {}, "SCREENING", "rule 4: ground class not known"),
    ("basin_subsidence", "multi_year_rate", {"ground": "suburban", "measured_rate_mm_yr": 30.0}, "VALIDATED",
     "a LOS rate is not compared with the vertical ceiling"),
    ("basin_subsidence", "multi_year_rate", {"ground": "suburban", "measured_vertical_mm_yr": 30.0}, "SCREENING",
     "rule 3: above 20 mm/yr vertical (catalogue section 2)"),
    ("basin_subsidence", "multi_year_rate", {"ground": "suburban", "expected_max_rate_mm_yr": 40.0}, "REFUSED",
     "rule 3: expected vertical rate above B2's ceiling"),
    ("basin_subsidence", "multi_year_rate", {"ground": "irrigated_farmland", "expected_max_rate_mm_yr": 40.0},
     "REFUSED", "the B2 envelope still applies on irrigated farmland"),
    ("basin_subsidence", "multi_year_rate", {"ground": "irrigated_farmland", "measured_vertical_mm_yr": 30.0},
     "SCREENING", "the B2 envelope still applies on irrigated farmland"),
    ("basin_subsidence", "multi_year_rate", {"ground": "suburban", "route": "disp"}, "SCREENING", "rule 2"),
    ("basin_subsidence", "multi_year_total", {"ground": "irrigated_farmland"}, "SCREENING",
     "rule 6: the transfer names only multi-year rates"),
    ("basin_subsidence", "series", {"ground": "suburban"}, "SCREENING", "rule 6"),
    ("basin_subsidence", "single_year_rate", {}, "REFUSED", "B1_B3 FAIL"),
    ("basin_subsidence", "feature_search", {"ground": "suburban"}, "SCREENING", "smaller footprint"),
    ("basin_subsidence", "rising_ground", {"ground": "suburban"}, "SCREENING", "sign not covered"),
    ("basin_subsidence", "trend_shape", {}, "RESEARCH", "no evidence"),
    ("infrastructure", "multi_year_rate", {"ground": "suburban"}, "SCREENING", "rule 6: no transfer (waits for V1)"),
    ("infrastructure", "profile", {"ground": "suburban"}, "SCREENING", "rule 6 and footprint"),
    ("infrastructure", "values_near_points", {}, "SCREENING", ""),
    ("infrastructure", "before_after", {}, "SCREENING", ""),
    ("oil_gas", "multi_year_rate_vertical", {"tracks": 2}, "SCREENING", "rule 6"),
    ("oil_gas", "multi_year_rate", {"regime": "fast_zone"}, "REFUSED", "inside a fast zone"),
    ("oil_gas", "values_near_points", {}, "SCREENING", ""),
    ("ccus_baseline", "multi_year_rate", {}, "SCREENING", "baseline only"),
    ("salt_caverns", "cavern_peak", {}, "SCREENING", ""),
    ("salt_caverns", "multi_year_rate", {"regime": "sinkhole_edge"}, "REFUSED", "sinkhole-prone ground"),
    ("coastal", "up_combined", {"tracks": 2}, "SCREENING", "rule 6"),
    ("coastal", "multi_year_rate_vertical", {}, "SCREENING", ""),
    ("saltwater_disposal", "multi_year_rate_vertical", {"tracks": 2}, "RESEARCH", "not offered"),
]


@pytest.mark.parametrize("skey,okey,kw,tier,why", NEW_STUDY_TIERS,
                         ids=[f"{a}-{b}-{i}" for i, (a, b, _, _, _) in enumerate(NEW_STUDY_TIERS)])
def test_new_study_tiers_match_the_catalogue(skey, okey, kw, tier, why):
    s = domains.study(skey)
    got, reasons = evidence.output_tier(s, s.output(okey), EV, **kw)
    assert got == tier, f"{skey}.{okey} {kw}: {got} ({reasons}); expected {tier}: {why}"


def test_every_new_study_output_is_screening_or_lower_without_a_transfer():
    for s in domains.STUDIES:
        if s.grandfathered or s.key in EV.transfers:
            continue
        for o in s.outputs:
            for g in (None, "suburban", "irrigated_farmland"):
                t = evidence.output_tier(s, o, EV, tracks=2, ground=g)[0]
                assert t in ("SCREENING", "RESEARCH", "REFUSED"), (s.key, o.key, g, t)


def test_the_offered_studies_are_the_first_slice():
    """D2: the default, groundwater and five screening studies; the rest are listed, not offered."""
    offered = {s.key for s in domains.STUDIES if s.builder_status == "offered"}
    assert offered == {"ground_subsidence", "basin_subsidence", "infrastructure", "oil_gas", "ccus_baseline",
                       "salt_caverns", "coastal"}
    for s in domains.STUDIES:
        assert all(o.sign != "up" or evidence.output_tier(s, o, EV)[0] in ("SCREENING", "RESEARCH", "REFUSED")
                   for o in s.outputs), f"{s.key}: rising ground waits for Gate 2"


def test_a_card_says_what_a_study_will_and_will_not_give():
    lines = evidence.card(domains.study("basin_subsidence"), EV)
    text = "\n".join(lines)
    assert "VALIDATED on suburban ground" in text and "INDICATIVE on irrigated farmland ground" in text
    assert "SCREENING elsewhere" in text
    assert "Cannot tell you:" in text and "(physics)" in text and "Set at creation: site_profile: groundwater_basin" in text
    coast = "\n".join(evidence.card(domains.study("coastal"), EV))
    assert "Both viewing directions required" in coast and "kind tide_gauges" in coast
    assert "Not offered yet" in "\n".join(evidence.card(domains.study("volcanoes"), EV))
    assert "this project" not in coast                       # no project: no track count
    assert "(this project: 1)" in "\n".join(evidence.card(domains.study("coastal"), EV, tracks=1))
    seis = "\n".join(evidence.card(domains.study("induced_seismicity"), EV))
    assert "does not place a well in any response tier." in seis
    for key in ("oil_gas", "salt_caverns"):
        assert "opposite signs" in "\n".join(evidence.card(domains.study(key), EV)), key


def test_insar_studies_prints_every_card(capsys):
    from insar_pipeline import cli
    assert cli.studies_main([]) == 0
    out = capsys.readouterr().out
    assert all(s.label in out for s in domains.STUDIES)
    assert cli.studies_main(["no_such_study"]) == 2


def test_every_default_output_is_in_the_expected_table():
    assert {k for k, _, _, _ in EXPECTED_TIERS} == {o.key for o in D.outputs}


def test_route_caps_cover_every_backend():
    assert set(config.ROUTE_TAG) == set(config.BACKENDS)
    assert set(config.ROUTE_TAG.values()) <= set(evidence.ROUTE_TAG_CAP)


def test_a_signed_override_applies_and_a_pending_one_does_not():
    ew, lc = D.output("east_west"), D.output("local_change")
    assert evidence.output_tier(D, ew, EV, tracks=2)[0] == "SCREENING"
    signed = replace(EV, overrides={k: dict(v, signed="test") for k, v in EV.pending_overrides.items()})
    assert evidence.output_tier(D, ew, signed, tracks=2)[0] == "INDICATIVE"
    assert evidence.output_tier(D, lc, signed)[0] == "SCREENING"
    # an override replaces the entry's grade only: the later caps still apply
    assert evidence.output_tier(D, ew, signed, tracks=1)[0] == "INDICATIVE"
    assert evidence.output_tier(D, ew, signed, tracks=2, route="disp")[0] == "SCREENING"
    assert evidence.output_tier(D, ew, signed, tracks=2, route="nisar")[0] == "RESEARCH"
    assert evidence.output_tier(D, ew, signed, tracks=2, measured_rate_mm_yr=50.0)[0] == "SCREENING"


def test_a_new_study_needs_a_transfer_and_matching_ground():
    basin = replace(D, key="basin_subsidence", grandfathered=False)
    rate = D.output("multi_year_rate")
    # the signed transfer names multi_year_rate from B2 and B1_B3; ground suburban is on B2
    assert evidence.output_tier(basin, rate, EV, ground="suburban")[0] == "VALIDATED"
    assert evidence.output_tier(basin, rate, EV, ground="dense_urban")[0] == "SCREENING"      # rule 4
    other = replace(D, key="coastal", grandfathered=False)
    assert evidence.output_tier(other, rate, EV, ground="suburban")[0] == "SCREENING"         # rule 6
    # the grandfathered default ignores rules 4 and 6
    assert evidence.output_tier(D, rate, EV, ground="dense_urban")[0] == "VALIDATED"
    # an output resting on both transfer ids is graded on the entry for the project's
    # ground (rule 4): B2 PASS on suburban, B1_B3 HELD on irrigated farmland; B2 alone
    # has no irrigated-farmland evidence
    both = replace(rate, evidence=("B2", "B1_B3"))
    assert evidence.output_tier(basin, both, EV, ground="suburban")[0] == "VALIDATED"
    assert evidence.output_tier(basin, both, EV, ground="irrigated_farmland")[0] == "INDICATIVE"
    assert evidence.output_tier(basin, rate, EV, ground="irrigated_farmland")[0] == "SCREENING"


def test_a_fail_refuses_only_inside_its_own_regime():
    series = replace(D.output("series"), evidence=("A",))
    assert evidence.output_tier(D, series, EV, regime="active_longwall")[0] == "REFUSED"
    assert evidence.output_tier(D, replace(series, footprint="area"), EV)[0] == "SCREENING"   # A failed at panels over longwall


# ------------------------------------------------------------------- config keys
def _cfg(**kw):
    return config.PipelineConfig(project_dir=Path("/tmp/x"), bbox=(-87.7, 38.3, -87.6, 38.4),
                                 start_date="2020-01-01", end_date="2024-12-31", **kw)


def test_a_config_without_the_study_keys_behaves_as_before():
    cfg = _cfg()
    assert cfg.study == "ground_subsidence" and cfg.study_registry_version == 0
    assert domains.study_of(cfg) is D
    assert not [p for p in config.validate_config(cfg) if "study" in p]


def test_an_unknown_study_or_a_newer_registry_is_refused():
    assert any("study = 'no_such_study'" in p for p in config.validate_config(_cfg(study="no_such_study")))
    newer = _cfg(study_registry_version=domains.REGISTRY_VERSION + 1)
    assert any("study_registry_version" in p for p in config.validate_config(newer))


def test_feature_layers_name_only_the_study_kinds(tmp_path):
    good = tmp_path / "wells.geojson"
    good.write_text('{"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, '
                    '"geometry": {"type": "Point", "coordinates": [-87.65, 38.35]}}]}')
    cfg = _cfg(study="oil_gas", feature_layers=(f"wells:{good}",))
    assert not [p for p in config.validate_config(cfg) if "feature_layers" in p]
    assert not [p for p in config.validate_inputs(cfg) if "feature_layers" in p]
    assert domains.layer_files(domains.study("oil_gas"), cfg) == [("wells", str(good))]
    bad = config.validate_config(_cfg(study="oil_gas", feature_layers=(f"caverns:{good}", "nocolon")))
    assert any("'caverns' is not a layer of the 'oil_gas' study" in p for p in bad)
    assert any("expected 'kind:path'" in p for p in bad)
    assert any("does not exist" in p for p in
               config.validate_inputs(_cfg(study="oil_gas", feature_layers=("wells:/no/such.geojson",))))
    assert any("relative path" in p for p in
               config.validate_inputs(_cfg(study="oil_gas", feature_layers=("wells:rel.geojson",))))
    assert any("Windows path" in p for p in
               config.validate_inputs(_cfg(study="oil_gas", feature_layers=("wells:C:\\data\\wells.geojson",))))


def test_a_study_not_offered_runs_only_as_an_experiment():
    assert any("not offered for client projects" in p for p in config.validate_config(_cfg(study="volcanoes")))
    assert not [p for p in config.validate_config(_cfg(study="volcanoes", purpose="experiment")) if "study" in p]
    assert not [p for p in config.validate_config(_cfg(study="coastal")) if "study" in p]


def test_write_project_applies_a_study_presets_once(tmp_path):
    import yaml
    aoi = areas.box(38.30, -87.70, 38.34, -87.64, "site")
    kw = dict(track=19, direction="ascending", report_areas=[("All", "full")], disp_frame="04861",
              projects_root=tmp_path / "p", configs_dir=tmp_path / "c")
    out = areas.write_project("Basin Test", aoi, "disp", "2020-01-01", "2024-12-31", study="basin_subsidence", **kw)
    raw = yaml.safe_load(out.read_text())
    assert raw["study"] == "basin_subsidence" and raw["site_profile"] == "groundwater_basin"
    assert "held from release" in out.read_text().splitlines()[0:12].__str__()
    cfg = config.load_config(str(out))
    assert cfg.study == "basin_subsidence" and cfg.site_profile == "groundwater_basin"
    with pytest.raises(ValueError, match="not offered"):
        areas.write_project("Volcano Test", aoi, "disp", "2020-01-01", "2024-12-31", study="volcanoes", **kw)
    with pytest.raises(ValueError, match="no study"):
        areas.write_project("Nope Test", aoi, "disp", "2020-01-01", "2024-12-31", study="nope", **kw)


def test_release_holds_every_study_but_the_default(tmp_path):
    from insar_pipeline import release
    rep = {"areas": [], "features": [], "route": "official"}
    codes = lambda c: {b["code"] for b in release.blockers(c, rep)}       # noqa: E731
    assert "study_not_released" not in codes(_cfg())
    assert "study_not_released" in codes(_cfg(study="basin_subsidence"))
    # a folder-only call reads the study from the report
    assert "study_not_released" in {b["code"] for b in release.blockers(str(tmp_path), dict(rep, study="coastal"))}
    # a config edited back to the default does not release another study's report
    assert "study_not_released" in {b["code"] for b in release.blockers(_cfg(), dict(rep, study="coastal"))}


def test_write_project_stamps_the_study_and_registry_version(tmp_path):
    import yaml
    aoi = areas.box(38.30, -87.70, 38.34, -87.64, "site")
    out = areas.write_project("Stamp Test", aoi, "disp", "2020-01-01", "2024-12-31", track=19, direction="ascending",
                              report_areas=[("All", "full")], disp_frame="04861",
                              projects_root=tmp_path / "p", configs_dir=tmp_path / "c")
    raw = yaml.safe_load(out.read_text())
    assert raw["study"] == "ground_subsidence" and raw["study_registry_version"] == domains.REGISTRY_VERSION
    cfg = config.load_config(str(out))
    assert cfg.study == "ground_subsidence" and cfg.study_registry_version == domains.REGISTRY_VERSION


# ------------------------------------------------------------------------- names
REGISTRY_FILES = ("evidence.yaml", "domains.py", "evidence.py", "VALIDATION.md", "LIMITATIONS.md", "CHANGELOG.md",
                  "README.md", "report_insights.py", "client_report.py", "areas.py", "config.py")
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".html", ".js", ".css", ".json", ".sh", ".toml", ".cfg",
                 ".csv", ".geojson", ".cmd", ".ps1", ".vbs", ".patch", ""}
# (file, digest) a literature citation may keep: a cited author in a reference list
CITED = {("research/atmospheric_temperature/ASSESSMENT.md",
          "5e2f91e6b4290d0b291891a5fc7f1222463ab45e87c80e59fa30608509b168b5")}
MAX_NAME_TOKENS = 6


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^0-9a-z]+", " ", s).strip()


def _tracked_text_files() -> list[str]:
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(PKG), "ls-files", "-z"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        out = None
    files = [f for f in (out.stdout.split("\0") if out and out.returncode == 0 else []) if f]
    files = sorted(set(files) | set(REGISTRY_FILES))         # the new files before their first commit
    return [f for f in files if Path(f).suffix.lower() in TEXT_SUFFIXES and (PKG / f).is_file()]


def test_no_dossier_name_in_any_tracked_file():
    """D13: author, operator and vendor names from the domain dossiers stay out
    of tracked files. The list is hashed (sha256 of the normalised name) so the
    test itself names no one; names are matched across line breaks."""
    hashes = set(need(PKG / "tests" / "data" / "denied_names.sha256").read_text().split())
    assert hashes and all(len(h) == 64 for h in hashes)
    hits = set()
    for f in _tracked_text_files():
        try:
            toks = _norm((PKG / f).read_text(encoding="utf-8")).split()
        except UnicodeDecodeError:
            continue
        for i in range(len(toks)):
            for k in range(1, MAX_NAME_TOKENS + 1):
                if i + k <= len(toks):
                    h = hashlib.sha256(" ".join(toks[i:i + k]).encode()).hexdigest()
                    if h in hashes and (f, h) not in CITED:
                        hits.add(f)
    assert not hits, f"a denied name is in {sorted(hits)}"


def test_the_name_check_finds_a_wrapped_name(tmp_path, monkeypatch):
    """The matcher itself: a listed two-word name split over two lines is found."""
    toks = _norm("Red Trail\nEnergy filed a plan").split()
    grams = {" ".join(toks[i:i + k]) for i in range(len(toks)) for k in range(1, 4) if i + k <= len(toks)}
    hashes = set((PKG / "tests" / "data" / "denied_names.sha256").read_text().split())
    assert any(hashlib.sha256(g.encode()).hexdigest() in hashes for g in grams)


def test_no_client_marker_in_the_registry_files():
    """The local marker file (scripts/check_repo_clean.sh) as whole words; skipped
    where it does not exist (CI, a fresh clone)."""
    p = Path(os.environ.get("INSAR_CLIENT_MARKERS", Path.home() / ".config" / "insar" / "client_markers.txt"))
    markers = [_norm(m.split("#", 1)[0]) for m in need(p).read_text().splitlines()]
    markers = [m for m in markers if m]
    hits = []
    # the files S0.4 adds or tags (older modules hold public place names in examples)
    for f in ("evidence.yaml", "domains.py", "evidence.py", "VALIDATION.md", "LIMITATIONS.md", "tests/test_domains.py"):
        text = " " + " ".join(_norm(line) for line in (PKG / f).read_text(encoding="utf-8").splitlines()) + " "
        hits += [f"{f}: marker {i + 1}" for i, m in enumerate(markers) if f" {m} " in text]
    assert not hits, hits
