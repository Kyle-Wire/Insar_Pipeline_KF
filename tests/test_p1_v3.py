"""
Phase 1, V3 (review_2026-10-08/PLAN.md section 3): experiment mode and the
QA site profile.

  purpose: experiment   the products stage runs the deliverables, the QA,
                        the figures and the validation harness (when
                        validation_protocol is set); never the GNSS tie,
                        client report, KMZ or release status.
  site_profile          which quality flags the ground explains; replaces
                        quality_assessment.EXPECTED_ON_FARMLAND. "farmland"
                        (default) is byte-for-byte today's behaviour.

Synthetic projects only, no network. Each test fails on the code before V3
(checked in a scratch copy with the pre-phase-1 config.py, pipeline.py and
quality_assessment.py).

Run: conda run -n insar python -m pytest tests/test_p1_v3.py -q
"""
import json
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from insar_pipeline import notify, pipeline, procs, progress, runs   # noqa: E402
from insar_pipeline import quality_assessment as qa                   # noqa: E402
from insar_pipeline import stack_qc                                   # noqa: E402
from insar_pipeline.config import ConfigError, load_config            # noqa: E402
import test_fix_L8 as L8                                              # noqa: E402  (its synthetic project)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("the QA tried to fetch OSM water in a test")
    monkeypatch.setattr(stack_qc, "fetch_osm_water", boom, raising=False)
    monkeypatch.setenv("INSAR_NOTIFY", "off")
    monkeypatch.setattr(notify, "LOGFILE", tmp_path / "notifications.log")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    yield
    procs.clear_stopping()


def _yaml(tmp_path, extra: str = "") -> Path:
    f = tmp_path / "c.yaml"
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n" + extra)
    return f


# ====================================================================== config
def test_config_defaults_are_todays_behaviour(tmp_path):
    cfg = load_config(str(_yaml(tmp_path)))
    assert (cfg.purpose, cfg.site_profile, cfg.validation_protocol) == ("client", "farmland", "")
    from insar_pipeline import config
    assert not config.is_experiment(cfg) and config.site_profile(cfg) == "farmland"


def test_config_reads_purpose_profile_and_protocol(tmp_path):
    proto = tmp_path / "protocol_B1.yaml"
    proto.write_text("experiment: B1\n")
    cfg = load_config(str(_yaml(tmp_path, f"purpose: experiment\nsite_profile: groundwater_basin\n"
                                          f"validation_protocol: {proto}\n")))
    from insar_pipeline import config
    assert config.is_experiment(cfg) and pipeline.is_experiment(cfg)
    assert config.site_profile(cfg) == "groundwater_basin" and cfg.validation_protocol == str(proto)


@pytest.mark.parametrize("line,words", [
    ("purpose: research\n", r"purpose = 'research'; expected one of \['client', 'experiment'\]"),
    ("site_profile: desert\n", r"site_profile = 'desert'; expected one of \['farmland', 'urban', 'mixed', "
                               r"'mine_longwall', 'groundwater_basin'\]"),
])
def test_config_refuses_an_unknown_purpose_or_profile(tmp_path, line, words):
    with pytest.raises(ConfigError, match=words):
        load_config(str(_yaml(tmp_path, line)))


def test_config_validation_protocol_needs_experiment_an_absolute_path_and_the_file(tmp_path):
    proto = tmp_path / "protocol_A.yaml"
    proto.write_text("experiment: A\n")
    with pytest.raises(ConfigError, match="purpose is not 'experiment'"):
        load_config(str(_yaml(tmp_path, f"validation_protocol: {proto}\n")))
    with pytest.raises(ConfigError, match="relative path"):
        load_config(str(_yaml(tmp_path, "purpose: experiment\nvalidation_protocol: protocol_A.yaml\n")))
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(str(_yaml(tmp_path, f"purpose: experiment\nvalidation_protocol: {tmp_path / 'nope.yaml'}\n")))


def test_config_and_qa_name_the_same_profiles():
    from insar_pipeline import config
    assert tuple(qa.SITE_PROFILES) == config.SITE_PROFILES
    assert config.PURPOSES == ("client", "experiment")
    assert set(qa.CHECKS) == set(qa.EXPECTED_ON_FARMLAND) and set(qa.CHECKS) == set(qa.PLAIN)
    for name, p in qa.SITE_PROFILES.items():
        assert set(p["expected"]) <= set(qa.CHECKS), name
        assert p["handled"] == (name == "farmland"), name          # only farmland has been checked
    # the old table, unchanged, under its old name
    assert qa.EXPECTED_ON_FARMLAND == {"reference_stability": True, "rate_precision": True,
                                       "distributed_scatterer_bias": True, "single_epoch_noise": False,
                                       "coverage": False, "network": False, "stitch": False}


# ====================================================================== QA: site profile
def test_farmland_profile_writes_the_record_it_always_wrote(tmp_path):
    a = qa.evaluate(L8.build(tmp_path / "a"))
    b = qa.evaluate(L8.build(tmp_path / "b", site_profile="farmland", control_ground_label="pasture"))
    for r in (a, b):
        assert "site_profile" not in r and "purpose" not in r
        assert all("on_farmland_expected" in c and "expected_on_site" not in c for c in r["checks"].values())
        assert "site_profile" not in r["send_rule"] and "experiment" not in r["send_rule"]
    # the words (B2, 28 Oct 2026): a control_ground_label someone set wins on
    # the farmland profile too; unset, the measured land cover - and this
    # synthetic project has no land-cover map, so plain "ground". The words
    # are the only difference between the two records.
    assert a["comparison_ground"]["words"] == "ground" and a["comparison_ground"]["source"] == "unmeasured"
    assert b["comparison_ground"]["words"] == "pasture" and b["comparison_ground"]["source"] == "config"
    strip = (lambda r: re.sub(r"/tmp\S*", "", json.dumps({k: v for k, v in r.items() if k != "comparison_ground"},
                                                         default=str)))
    assert "pasture" not in strip(a) and "AOI pasture, 2 km buffer" in strip(b)
    assert strip(a) == strip(b).replace("pasture", "ground").replace("Pasture", "Ground")
    # the profile's own phrase names the profile
    assert "expected on farmland" in qa.send_rule(_flagged(a, "rate_precision"))["notes"][0]


def _flagged(rec, *checks):
    """An evaluated record cut to its checks' site annotations, every check
    passing but `checks`, and nothing else that could hold the report."""
    keep = ("on_farmland_expected", "expected_on_site", "expected_handled", "expected_why")
    r = {k: rec[k] for k in ("site_profile", "purpose") if k in rec}
    r["checks"] = {n: {**{k: c[k] for k in keep if k in c}, "pass": n not in checks} for n, c in rec["checks"].items()}
    r["geolocation"] = {"flag": False}
    return r


def test_longwall_profile_never_says_farmland_is_expected(tmp_path):
    r = qa.evaluate(L8.build(tmp_path, site_profile="mine_longwall", control_ground_label="pasture"))
    txt = json.dumps(r, default=str)
    assert "expected on farmland" not in txt and "farmland-expected" not in txt
    assert not any("on_farmland_expected" in c for c in r["checks"].values())
    sp = r["site_profile"]
    assert sp["name"] == "mine_longwall" and sp["ground"] == "pasture"
    assert sp["expected"] == ["rate_precision", "distributed_scatterer_bias"] and sp["handled"] is False
    ch = r["checks"]
    assert ch["rate_precision"]["expected_on_site"] is True and ch["rate_precision"]["expected_handled"] is False
    assert ch["reference_stability"]["expected_on_site"] is False and "expected_why" not in ch["reference_stability"]
    assert "pasture" in ch["distributed_scatterer_bias"]["plain"] and "farmland" not in ch["rate_precision"]["plain"]
    # prose says the comparison ground; keys other modules read do not change
    assert "farmland_spread_1sigma_mm_yr" in txt or "farmland" not in txt
    for s in _strings(r):
        if re.search(r"\bfarmland\b", s):
            assert "checked on farmland" in s, s
    sr = qa.send_rule(_flagged(r, "rate_precision"))
    assert sr["value"] == "hold for specialist" and sr["site_profile"] == "mine_longwall"
    assert "expected at a longwall mining site" in sr["hold_reasons"][0]
    assert "only been checked on farmland" in sr["hold_reasons"][0]
    assert "farmland-expected" not in sr["rule"] and "thin pasture" in sr["rule"]
    assert "site profile: mine_longwall" in qa.format_report(r)


def _strings(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def _record(profile, ground="open ground", **checks):
    """A minimal QA record as annotate_site leaves it."""
    rec = {"checks": {c: {"pass": checks.get(c, True)} for c in qa.CHECKS}, "geolocation": {"flag": False}}
    cfg = types.SimpleNamespace(site_profile=profile, control_ground_label=ground, purpose="client")
    return qa.annotate_site(cfg, rec)


def test_groundwater_basin_holds_on_whole_area_motion_and_names_why():
    clean = _record("groundwater_basin", ground="farmland")
    sr = qa.send_rule(clean)
    # nothing flagged, but the far field moves with the signal: relative figures understate it
    assert sr["value"] == "hold for specialist"
    assert any("whole area moves in a subsiding groundwater basin" in h for h in sr["hold_reasons"])
    ref = qa.send_rule(_record("groundwater_basin", ground="farmland", reference_stability=False))
    assert ref["hold_reasons"][0].startswith("reference stability flagged (expected in a subsiding groundwater basin")
    # the same flag on farmland is the Site A "send with notes"
    farm = {"checks": {c: {"pass": c != "reference_stability"} for c in qa.CHECKS}, "geolocation": {"flag": False}}
    assert qa.send_rule(farm)["value"] == "send with notes"


def test_mixed_and_urban_profiles_hold_on_flags_the_ground_does_not_explain():
    mixed = qa.send_rule(_record("mixed", rate_precision=False, coverage=False))
    assert mixed["value"] == "hold for specialist"
    assert any(h.startswith("rate precision flagged (expected on mixed built-up and vegetated ground")
               for h in mixed["hold_reasons"])
    assert "coverage flagged" in mixed["hold_reasons"]
    urban = qa.send_rule(_record("urban", distributed_scatterer_bias=False))
    assert urban["hold_reasons"] == ["distributed scatterer bias flagged"]
    assert qa.send_rule(_record("urban"))["value"] == "send"


def test_a_profile_whose_coping_has_been_checked_turns_expected_flags_into_notes(monkeypatch):
    monkeypatch.setitem(qa.SITE_PROFILES, "mixed", {**qa.SITE_PROFILES["mixed"], "handled": True})
    sr = qa.send_rule(_record("mixed", rate_precision=False))
    assert sr["value"] == "send with notes"
    assert sr["notes"] == ["rate precision flagged (expected on mixed built-up and vegetated ground; the figures "
                           "are built to cope with it)"]
    assert "send with notes: only flags expected on mixed built-up and vegetated ground" in sr["rule"]
    assert "until the figures' way of coping" not in sr["rule"]
    basin = qa.send_rule(_record("groundwater_basin"))["rule"]
    assert "also a flag the ground explains" in basin and "and the whole area moving with the signal" in basin
    assert "on this site profile" not in qa.send_rule(_record("urban"))["rule"]


def test_send_rule_says_the_comparison_ground_on_other_profiles():
    rec = _record("mine_longwall", ground="pasture")
    rec["far_field"] = {"thin": True}
    rec["area_testability"] = {"North": {"kind": "area", "testable": False, "not_testable_reason": "too small"}}
    sr = qa.send_rule(rec)
    assert "too little ordinary pasture to test the report areas against" in sr["hold_reasons"]
    assert "report area 'North' cannot be compared with pasture: too small" in sr["hold_reasons"]
    assert not any(re.search(r"\bfarmland\b", x) for x in sr["reasons"])


def test_send_rule_on_an_unknown_profile_expects_nothing():
    rec = {"checks": {"rate_precision": {"pass": False}}, "geolocation": {"flag": False},
           "site_profile": {"name": "arid", "where": "on arid ground", "ground": "desert"}}
    sr = qa.send_rule(rec)
    assert sr["hold_reasons"] == ["rate precision flagged"]


def test_reword_ground_keeps_identifiers_paths_and_farmland_grounds():
    assert qa.reword_ground("ordinary farmland; Farmland null", "pasture") == "ordinary pasture; Pasture null"
    assert qa.reword_ground("farmland_spread_1sigma_mm_yr (farmland)", "pasture") == \
        "farmland_spread_1sigma_mm_yr (pasture)"
    assert qa.reword_ground("/data/farmland/x.geojson", "pasture") == "/data/farmland/x.geojson"
    assert qa.reword_ground("ordinary farmland", "irrigated farmland") == "ordinary farmland"
    assert qa.reword_ground(3.0, "pasture") == 3.0


# ====================================================================== QA: purpose
def test_experiment_purpose_is_recorded_and_said(tmp_path):
    r = qa.evaluate(L8.build(tmp_path, purpose="experiment"))
    assert r["purpose"] == "experiment"
    assert r["send_rule"]["experiment"] == qa.EXPERIMENT_SEND_NOTE and r["send_rule"]["value"] in qa.SEND_VALUES
    assert "EXPERIMENT RUN" in qa.format_report(r).splitlines()[1]


# ====================================================================== pipeline
def _cfg(tmp_path, backend="miaplpy", **attrs):
    cfg = load_config(str(_yaml(tmp_path)))
    cfg.backend = backend
    for k, v in attrs.items():
        setattr(cfg, k, v)
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _stubs(monkeypatch, calls, validation_ok=True):
    def mk(name):
        def f(cfg):
            calls.append(name)
            progress.emit(cfg.project_dir, name, "done")
            return True
        return f

    def export(cfg):
        calls.append("deliverables")
        return cfg.project_dir / "deliverables"
    monkeypatch.setattr(pipeline, "export_all", export)
    monkeypatch.setattr(pipeline, "geocode_missing_layers", lambda cfg: {})
    monkeypatch.setattr(pipeline, "run_quality_assessment", mk("quality_assessment"))
    monkeypatch.setattr(pipeline, "run_figures", mk("figures"))
    monkeypatch.setattr(pipeline, "run_client_report", mk("client_report"))

    def validate(cfg):
        calls.append(("validation", cfg.validation_protocol))
        if not validation_ok:
            raise RuntimeError("protocol hash does not match")
        return cfg.project_dir / "validation"
    monkeypatch.setitem(sys.modules, "insar_pipeline.validation", types.SimpleNamespace(run=validate))
    from insar_pipeline import provenance, release

    def no_release(*a, **k):
        raise AssertionError("an experiment run asked for the release status")
    monkeypatch.setattr(release, "status", no_release)
    monkeypatch.setattr(provenance, "record_stages", lambda cfg: calls.append("provenance") or
                        cfg.project_dir / "deliverables" / "provenance.json")


def _old_client_files(cfg):
    D = cfg.project_dir / "deliverables"
    D.mkdir(parents=True, exist_ok=True)
    (D / "client_report.html").write_text("a report written before V3")
    (D / "client_report.json").write_text("{}")
    (D / "Exp_ground_movement_DRAFT.kmz").write_text("kmz")
    return D


def test_experiment_products_run_validates_and_builds_no_client_report(tmp_path, monkeypatch):
    proto = tmp_path / "protocol_B1.yaml"
    proto.write_text("experiment: B1\n")
    cfg = _cfg(tmp_path, purpose="experiment", validation_protocol=str(proto))
    D = _old_client_files(cfg)
    calls = []
    _stubs(monkeypatch, calls)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    assert calls == ["deliverables", "quality_assessment", "figures", ("validation", str(proto)), "provenance"]
    st = progress.latest(cfg.project_dir)
    assert st["client_report"]["status"] == "skipped" and st["validation"]["status"] == "done"
    assert not (D / "client_report.html").exists() and not list(D.glob("*ground_movement*.kmz"))
    kept = list((D / "experiment_not_for_clients").glob("*/*"))
    assert sorted(p.name for p in kept) == ["Exp_ground_movement_DRAFT.kmz", "client_report.html", "client_report.json"]
    assert "NOT FOR CLIENTS" in (D / pipeline.EXPERIMENT_NOTE).read_text()
    assert not (D / pipeline.SCREENING_NOTE).exists()
    r = json.loads((cfg.project_dir / runs.RESULTFILE).read_text())
    assert r["status"] == "done" and r["purpose"] == "experiment"
    assert "experiment run" in r["plain_cause"] and "3 older client file(s)" in r["plain_cause"]


def test_experiment_without_a_protocol_skips_validation_and_says_so(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, purpose="experiment")
    calls = []
    _stubs(monkeypatch, calls)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    assert "client_report" not in calls and not any(isinstance(c, tuple) for c in calls)
    assert progress.latest(cfg.project_dir)["validation"]["status"] == "skipped"
    r = json.loads((cfg.project_dir / runs.RESULTFILE).read_text())
    assert "no validation_protocol" in r["plain_cause"]


def test_a_failed_validation_fails_the_run(tmp_path, monkeypatch):
    proto = tmp_path / "p.yaml"
    proto.write_text("x: 1\n")
    cfg = _cfg(tmp_path, purpose="experiment", validation_protocol=str(proto))
    calls = []
    _stubs(monkeypatch, calls, validation_ok=False)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 1
    r = json.loads((cfg.project_dir / runs.RESULTFILE).read_text())
    assert r["failed_stages"] == ["validation"] and "client_report" not in calls
    assert progress.latest(cfg.project_dir)["validation"]["status"] == "failed"


def test_experiment_on_a_route_without_the_qa_still_validates(tmp_path, monkeypatch):
    proto = tmp_path / "p.yaml"
    proto.write_text("x: 1\n")
    cfg = _cfg(tmp_path, backend="hyp3", purpose="experiment", validation_protocol=str(proto))
    calls = []
    _stubs(monkeypatch, calls)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    assert calls == ["deliverables", ("validation", str(proto)), "provenance"]


def test_nisar_runs_the_qa_and_figures_too(tmp_path, monkeypatch):
    """26 Oct 2026 (Site E NISAR): the nisar route was the example of a
    route without the QA; it runs the QA and figures now, and as a client
    project the client report as well."""
    proto = tmp_path / "p.yaml"
    proto.write_text("x: 1\n")
    cfg = _cfg(tmp_path, backend="nisar", purpose="experiment", validation_protocol=str(proto))
    calls = []
    _stubs(monkeypatch, calls)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    assert calls == ["deliverables", "quality_assessment", "figures", ("validation", str(proto)), "provenance"]
    (tmp_path / "c").mkdir()
    cfg2 = _cfg(tmp_path / "c", backend="nisar")
    calls2 = []
    _stubs(monkeypatch, calls2)
    assert pipeline.run_from(cfg2, "products", watchdog=False, notify=False) == 0
    assert calls2 == ["deliverables", "quality_assessment", "figures", "client_report"]


def test_no_acceptance_experiment_sets_client_files_aside_without_validating(tmp_path, monkeypatch):
    proto = tmp_path / "p.yaml"
    proto.write_text("x: 1\n")
    cfg = _cfg(tmp_path, purpose="experiment", validation_protocol=str(proto))
    D = _old_client_files(cfg)
    calls = []
    _stubs(monkeypatch, calls)
    assert pipeline.run_from(cfg, "products", no_acceptance=True, watchdog=False, notify=False) == 0
    assert calls == ["deliverables"]
    assert not (D / "client_report.json").exists()
    assert progress.latest(cfg.project_dir)["validation"]["status"] == "skipped"


def test_client_purpose_keeps_the_client_report_chain(tmp_path, monkeypatch):
    """A guard, not a new behaviour: the default purpose runs the chain it always ran."""
    cfg = _cfg(tmp_path)
    calls = []
    _stubs(monkeypatch, calls)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    assert calls == ["deliverables", "quality_assessment", "figures", "client_report"]
    r = json.loads((cfg.project_dir / runs.RESULTFILE).read_text())
    assert "purpose" not in r
    assert "validation" not in progress.latest(cfg.project_dir)


def test_the_dashboards_report_rebuild_refuses_an_experiment(tmp_path, monkeypatch):
    """dashboard.rebuild_report calls pipeline.run_client_report directly."""
    cfg = _cfg(tmp_path, purpose="experiment")
    from insar_pipeline import client_report, gnss_tie

    def boom(*a, **k):
        raise AssertionError("an experiment reached the client report")
    monkeypatch.setattr(client_report, "build", boom)
    monkeypatch.setattr(gnss_tie, "run", boom)
    monkeypatch.setattr(pipeline, "record_report_stages", boom)
    assert pipeline.run_client_report(cfg) is False
    assert progress.latest(cfg.project_dir)["client_report"]["status"] == "skipped"


def test_set_aside_screening_unchanged_and_reasons_checked(tmp_path):
    cfg = _cfg(tmp_path, backend="disp")
    D = _old_client_files(cfg)
    moved = pipeline.set_aside_client_files(cfg)
    assert len(moved) == 3 and all("screening_not_for_clients" in str(m) for m in moved)
    assert "SCREENING ONLY" in (D / pipeline.SCREENING_NOTE).read_text()
    with pytest.raises(ValueError):
        pipeline.set_aside_client_files(cfg, "draft")


# ====================================================================== independent review (8 Oct)
def test_reword_ground_leaves_file_names_inside_a_sentence_and_backslashes_in_the_label():
    s = "could not read /data/farmland/x.geojson or farmland.geojson; ordinary farmland. Farmland null"
    assert qa.reword_ground(s, "pasture") == ("could not read /data/farmland/x.geojson or farmland.geojson; "
                                              "ordinary pasture. Pasture null")
    # a replacement string would read "\1" as a group reference and raise
    assert qa.reword_ground("thin farmland", r"grass\1scrub") == r"thin grass\1scrub"


def _harness_cfg(tmp_path, protocol_text="x: 1\n"):
    proto = tmp_path / "protocol_B1.yaml"
    proto.write_text(protocol_text)
    f = _yaml(tmp_path, f"purpose: experiment\nvalidation_protocol: {proto}\n")
    cfg = load_config(str(f))
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    return cfg, f, proto


def test_validation_stage_calls_the_harness_command_line_when_it_has_no_run(tmp_path, monkeypatch):
    """validation.py (the harness group's) has main(argv, cfg=None) and no
    run(cfg): the stage must call that, with the config file (the harness
    finds the protocol's run from its name) and the protocol."""
    cfg, f, proto = _harness_cfg(tmp_path)
    seen = []

    def main(argv=None, cfg=None):
        seen.append((list(argv), cfg))
        return 0
    monkeypatch.setitem(sys.modules, "insar_pipeline.validation", types.SimpleNamespace(main=main))
    assert pipeline.run_validation(cfg) is True
    assert seen == [([str(Path(f).resolve()), "--protocol", str(proto)], cfg)]
    assert progress.latest(cfg.project_dir)["validation"]["status"] == "done"


def test_validation_stage_fails_on_a_non_zero_harness_exit_or_no_entry_point(tmp_path, monkeypatch):
    cfg, _, _ = _harness_cfg(tmp_path)
    monkeypatch.setitem(sys.modules, "insar_pipeline.validation",
                        types.SimpleNamespace(main=lambda argv=None, cfg=None: 1))
    assert pipeline.run_validation(cfg) is False
    st = progress.latest(cfg.project_dir)["validation"]
    assert st["status"] == "failed" and "exit code 1" in st["msg"]
    monkeypatch.setitem(sys.modules, "insar_pipeline.validation", types.SimpleNamespace())
    assert pipeline.run_validation(cfg) is False
    assert "neither run(cfg) nor main" in progress.latest(cfg.project_dir)["validation"]["msg"]


def test_the_real_harness_has_an_entry_point_the_stage_can_call():
    """Integration guard with the harness group's module."""
    import inspect
    v = pytest.importorskip("insar_pipeline.validation")
    if callable(getattr(v, "run", None)):
        return
    params = inspect.signature(v.main).parameters
    assert "argv" in params and "cfg" in params
