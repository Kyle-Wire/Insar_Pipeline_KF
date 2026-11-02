"""16 Oct 2026, readiness review HND-03: configuration validation.

load_config refused unknown keys, duplicates, bad choices and impossible
ranges, but not wrong TYPES: the text 'no' in a true/false setting read as
True, a quoted '0.6' stayed text, 2.7 was kept for a count, and 'high' or
[0.4] for a number, a YAML syntax error or a list instead of settings raised
bare ValueError/TypeError/yaml errors that callers catching ConfigError
missed. stack_filter_strength 7 and gnss_tie_radius_km 40000 loaded. A
settings.json that did not parse read as no settings with nothing said, and
the next save() replaced it, losing its other keys.

Each test failed on the code at f3a0a0a. Configs and settings files in
tmp_path only (INSAR_SETTINGS points at a temporary file); no real project."""
import json
import logging

import pytest

from insar_pipeline import settings
from insar_pipeline.config import ConfigError, load_config

BASE_LINES = 4                     # the extra settings start on line BASE_LINES + 1


def _yaml(tmp_path, extra: str = "", name: str = "c.yaml"):
    """A clean config (classic route, no project files needed) plus `extra`."""
    p = tmp_path / name
    p.write_text(f"project_dir: {tmp_path / 'proj'}\n"
                 "bbox: [-100.2, 40.1, -100.0, 40.3]\n"
                 "start_date: 2020-01-01\n"
                 "end_date: 2021-01-01\n" + extra)
    return str(p)


def _refused(tmp_path, extra: str) -> str:
    with pytest.raises(ConfigError) as e:
        load_config(_yaml(tmp_path, extra))
    return str(e.value)


# ------------------------------------------------------------------ true/false settings
def test_true_false_words_read_as_what_they_say_quoted_or_not(tmp_path):
    """The recognised words (true/false, yes/no, on/off, 1/0, any case,
    quoted or not) give the bool they name; 'no' in quotes used to read as
    True. Text settings that carry the same words for ISCE/MintPy
    (stack_virtual_merge 'False', mintpy_deramp 'no') stay text."""
    falses = ["'no'", '"false"', "'Off'", "'0'", "0", "no", "False", "OFF", "nO"]
    trues = ["'yes'", '"TRUE"', "'on'", "'1'", "1", "yes", "True", "On", "yEs"]
    for word in falses:
        cfg = load_config(_yaml(tmp_path, f"skip_existing: {word}\n"))        # default True
        assert cfg.skip_existing is False, word
    for word in trues:
        cfg = load_config(_yaml(tmp_path, f"stack_use_gpu: {word}\n"))        # default False
        assert cfg.stack_use_gpu is True, word
    cfg = load_config(_yaml(tmp_path, "stack_virtual_merge: 'False'\nmintpy_deramp: 'no'\ndisp_frame_id: '04861'\n"
                                      "skip_existing: 'no'\n"))
    assert (cfg.stack_virtual_merge, cfg.mintpy_deramp, cfg.disp_frame_id) == ("False", "no", "04861")
    assert cfg.skip_existing is False


def test_a_value_that_is_not_true_or_false_is_refused_by_name_and_line(tmp_path):
    """Anything else in a true/false setting ('maybe', 2, 0.5) is a
    ConfigError naming the setting, its line and the value; 'maybe' used to
    load as True."""
    for value in ("maybe", "'nope'", "2", "0.5"):
        msg = _refused(tmp_path, f"subset_to_aoi: {value}\n")
        assert f"subset_to_aoi (line {BASE_LINES + 1})" in msg and "not true or false" in msg, msg


# ------------------------------------------------------------------ counts, numbers, lists
def test_counts_take_whole_numbers_only(tmp_path):
    """An int setting takes a whole number: 20.0 reads as 20; 2.7, the text
    'three', a quoted '20' and true are ConfigErrors (2.7 used to load as
    2.7, 'three' raised a bare ValueError)."""
    cfg = load_config(_yaml(tmp_path, "kmz_point_step: 20.0\n"))
    assert cfg.kmz_point_step == 20 and type(cfg.kmz_point_step) is int
    for value, words in (("2.7", "must be a whole number"), ("three", "must be a whole number"),
                         ("'20'", "remove the quotes"), ("true", "must be a whole number")):
        msg = _refused(tmp_path, f"kmz_point_step: {value}\n")
        assert "kmz_point_step (line 5)" in msg and words in msg, (value, msg)


def test_numbers_refuse_text_lists_and_empty_values(tmp_path):
    """A float setting takes a number: a quoted '0.6' (used to load as text),
    'high' and [0.4] (used to raise bare ValueError/TypeError), true and an
    empty value (used to load as None) are ConfigErrors. An unquoted 1e-3,
    which YAML 1.1 reads as text, is the number it says; an int stays that
    int. bbox must be numbers ('a' used to raise a bare ValueError)."""
    for value, words in (("'0.6'", "remove the quotes"), ("high", "must be a number"),
                         ("[0.4]", "must be a number"), ("true", "must be a number"), ("", "is empty")):
        msg = _refused(tmp_path, f"min_temporal_coherence: {value}\n")
        assert "min_temporal_coherence (line 5)" in msg and words in msg, (value, msg)
    cfg = load_config(_yaml(tmp_path, "stack_unw_min_component_fraction: 1e-3\nderamp_buffer_m: 2000\n"))
    assert cfg.stack_unw_min_component_fraction == 0.001
    assert cfg.deramp_buffer_m == 2000 and type(cfg.deramp_buffer_m) is int
    p = tmp_path / "b.yaml"
    p.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [a, 40.1, -100.0, 40.3]\n"
                 "start_date: 2020-01-01\nend_date: 2021-01-01\n")
    with pytest.raises(ConfigError, match=r"bbox \(line 2\).*must be numbers"):
        load_config(str(p))


def test_list_settings_take_lists(tmp_path):
    """A list setting given one text value is a ConfigError saying it must be
    a list (reference_gnss_stations: ABCD used to load, to be read letter by letter);
    an empty one is an empty list."""
    msg = _refused(tmp_path, "reference_gnss_stations: ABCD\n")
    assert "reference_gnss_stations (line 5)" in msg and "must be a list" in msg, msg
    cfg = load_config(_yaml(tmp_path, "exclude_pairs:\nreport_areas:\n"))
    assert cfg.exclude_pairs == () and cfg.report_areas == ()


# ------------------------------------------------------------------ ranges with a stated basis
def test_filter_strength_is_the_goldstein_alpha_0_to_1(tmp_path):
    """stack_filter_strength is the Goldstein-Werner alpha, defined on
    [0, 1]: 7 (a typo for 0.7) is refused, 0 and 1 load; 7 used to load."""
    msg = _refused(tmp_path, "stack_filter_strength: 7\n")
    assert "stack_filter_strength" in msg and "[0, 1]" in msg, msg
    for ok in ("0", "0.4", "1"):
        assert load_config(_yaml(tmp_path, f"stack_filter_strength: {ok}\n")).stack_filter_strength == float(ok)


def test_gnss_tie_radius_stays_within_the_regional_floor(tmp_path):
    """gnss_tie_radius_km is at most the 100 km whose stations set the tie's
    uncertainty floor (gnss_tie.REGIONAL_FLOOR_RADIUS_KM): 40000 is refused
    (it used to load), 0 (off), 40 and 100 load, -5 is still refused."""
    from insar_pipeline.gnss_tie import REGIONAL_FLOOR_RADIUS_KM
    assert REGIONAL_FLOOR_RADIUS_KM == 100.0
    msg = _refused(tmp_path, "gnss_tie_radius_km: 40000\n")
    assert "gnss_tie_radius_km" in msg and "100 km" in msg, msg
    _refused(tmp_path, "gnss_tie_radius_km: -5\n")
    for ok in ("0", "40", "100"):
        assert load_config(_yaml(tmp_path, f"gnss_tie_radius_km: {ok}\n")).gnss_tie_radius_km == float(ok)


# ------------------------------------------------------------------ the file itself
def test_malformed_yaml_is_a_config_error_naming_file_and_line(tmp_path):
    """A YAML syntax error is a ConfigError with the file and the line (it was
    a raw yaml ScannerError/ParserError); a file of anything but settings is
    a ConfigError too (a list of mappings raised a bare TypeError)."""
    p = tmp_path / "bad.yaml"
    p.write_text("project_dir: /x\nbbox: [-100.2, 40.1, -100.0, 40.3]\n  start_date: 2020-01-01\n")
    with pytest.raises(ConfigError) as e:
        load_config(str(p))
    assert str(p) in str(e.value) and "line 3" in str(e.value) and "not valid YAML" in str(e.value), str(e.value)
    p.write_text("bbox: [-100.2, 40.1, -100.0, 40.3\nstart_date: 2020-01-01\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(str(p))
    p.write_text("- project_dir: /x\n- bbox: [1, 2, 3, 4]\n")
    with pytest.raises(ConfigError, match="'name: value'"):
        load_config(str(p))


# ------------------------------------------------------------------ settings.json
@pytest.fixture
def settings_at(tmp_path, monkeypatch):
    f = tmp_path / "insar" / "settings.json"
    f.parent.mkdir()
    monkeypatch.setenv("INSAR_SETTINGS", str(f))
    monkeypatch.delenv("INSAR_PROJECTS", raising=False)
    return f


def test_unparsable_settings_warn_once_and_read_as_none(settings_at, caplog):
    """A settings.json that does not parse still reads as {} (callers keep
    working) but logs a WARNING naming the file and what to do - once, not on
    every call; it used to be silent. settings.problem() (insar doctor) says
    the same; a JSON list is not settings either."""
    settings_at.write_text('{"projects_root": "/data/projects", "firm": "Example Engineering",')
    caplog.set_level(logging.WARNING, logger="insar_pipeline.settings")
    assert settings.load() == {} and settings.load() == {}
    assert str(settings.projects_root()).endswith("insar_projects")
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and str(settings_at) in r.getMessage()]
    assert len(warns) == 1, [r.getMessage() for r in caplog.records]
    assert "not valid JSON" in warns[0].getMessage() and "Correct the file" in warns[0].getMessage()
    assert "not valid JSON" in settings.problem()
    settings_at.write_text('["firm"]')
    assert settings.load() == {} and "holds a JSON list" in settings.problem()


def test_save_never_overwrites_an_unparsable_settings_file(settings_at):
    """save() refuses (SettingsError) to replace a settings.json that does not
    parse, so its keys are not lost - it used to write the new key alone. Once
    the file parses, save() merges into it and the other keys survive; an
    empty file is no settings and is saved over."""
    broken = '{"projects_root": "/data/projects", "firm": "Example Engineering",'
    settings_at.write_text(broken)
    with pytest.raises(settings.SettingsError, match="Not overwriting"):
        settings.save(preparer="A. Engineer")
    assert settings_at.read_text() == broken
    settings_at.write_text('{"projects_root": "/data/projects", "firm": "Example Engineering"}')
    settings.save(preparer="A. Engineer")
    assert json.loads(settings_at.read_text()) == {"projects_root": "/data/projects", "firm": "Example Engineering",
                                                   "preparer": "A. Engineer"}
    assert settings.problem() is None and settings.identity()["preparer"] == "A. Engineer"
    settings_at.write_text("")
    settings.save(firm="Example Engineering")
    assert settings.load() == {"firm": "Example Engineering"}
