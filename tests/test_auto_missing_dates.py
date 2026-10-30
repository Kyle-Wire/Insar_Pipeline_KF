"""13 Oct 2026: catalogue-defect dates are accepted automatically and written
into the config; anything else still stops the run."""
from types import SimpleNamespace

import yaml

from insar_pipeline import data_acquisition as da


def test_defect_signatures():
    assert da.catalogue_defect("...\nValueError: max() arg is an empty sequence\nERROR conda") 
    assert da.catalogue_defect("ValueError: All bursts must have consecutive burst IDs. Found: [1, 1]")
    assert da.catalogue_defect("ConnectionError: Max retries exceeded") is None
    assert da.catalogue_defect("ASFAuthenticationError: 502 Bad Gateway") is None
    assert da.catalogue_defect("") is None


def test_dates_are_appended_to_the_config_with_a_reason(tmp_path):
    p = tmp_path / "proj.yaml"
    p.write_text("backend: miaplpy\nallow_missing_dates: ['20250101']\nend_date: '2026-06-30'\n")
    cfg = SimpleNamespace(_source_path=p, allow_missing_dates=("20250101",))
    out = da.record_auto_accepted(cfg, {"20260610": "burst annotations empty",
                                        "20250101": "x"})
    assert out == p
    doc = yaml.safe_load(p.read_text())
    assert doc["allow_missing_dates"] == ["20250101", "20260610"]
    assert doc["backend"] == "miaplpy" and doc["end_date"] == "2026-06-30"   # rest of the file kept
    assert "added automatically" in p.read_text() and "20260610 burst annotations empty" in p.read_text()
    assert p.read_text().count("allow_missing_dates:") == 1
    assert cfg.allow_missing_dates == ("20250101", "20260610")
    # appending again keeps one key and both dates
    da.record_auto_accepted(cfg, {"20260622": "y"})
    doc = yaml.safe_load(p.read_text())
    assert doc["allow_missing_dates"] == ["20250101", "20260610", "20260622"]


def test_no_config_file_means_this_run_only(tmp_path):
    cfg = SimpleNamespace(_source_path=None, allow_missing_dates=())
    assert da.record_auto_accepted(cfg, {"20260610": "x"}) is None
