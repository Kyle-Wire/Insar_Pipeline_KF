"""7 Oct 2026 integration: crosscheck.record_is_current, the one rule the
report, the QA and the release gate use to decide whether a cross_check.json
or injection_recovery.json record belongs to this run.

A record counts only when its option is on and it was generated no earlier
than the velocity product it checked (the geocoded one when there is one).
"""
import os
from datetime import datetime

import pytest

from insar_pipeline import crosscheck as cc
from insar_pipeline.config import PipelineConfig

OPTIONS = ("cross_check_disp", "injection_test")
# a fixed local time well away from a clock change: records are written in
# naive local time, which repeats an hour when the clocks go back
T = datetime(2026, 9, 15, 12, 0, 0).timestamp()


def _cfg(tmp_path, option=None):
    cfg = PipelineConfig(project_dir=tmp_path / "proj", bbox=[-87.52, 38.29, -87.47, 38.33],
                         start_date="2020-01-01", end_date="2022-12-31")
    for o in OPTIONS:
        setattr(cfg, o, o == option)
    return cfg


def _velocity(cfg, t, geo=False) -> None:
    """A fake MintPy velocity product with mtime t."""
    f = cfg.mintpy_dir / ("geo/geo_velocity.h5" if geo else "velocity.h5")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"")
    os.utime(f, (t, t))


def _rec(t) -> dict:
    """A record as crosscheck/injection_test write it (local time, seconds)."""
    return {"generated": datetime.fromtimestamp(t).isoformat(timespec="seconds")}


@pytest.mark.parametrize("option", OPTIONS)
def test_option_off_is_not_current(tmp_path, option):
    t = T
    cfg = _cfg(tmp_path, option=None)
    _velocity(cfg, t)
    assert cc.record_is_current(cfg, _rec(t + 600), option) is False
    # the other option being on does not make this one's record current
    other = next(o for o in OPTIONS if o != option)
    assert cc.record_is_current(_cfg(tmp_path, option=other), _rec(t + 600), option) is False
    assert cc.record_is_current(_cfg(tmp_path, option=option), _rec(t + 600), option) is True


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("rec", [None, {}])
def test_no_record_is_not_current(tmp_path, option, rec):
    cfg = _cfg(tmp_path, option)
    _velocity(cfg, T)
    assert cc.record_is_current(cfg, rec, option) is False


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("generated", [None, "", "not a date", "2026-13-40T99:00:00", 12345])
def test_unreadable_generated_is_not_current(tmp_path, option, generated):
    cfg = _cfg(tmp_path, option)
    _velocity(cfg, T)
    assert cc.record_is_current(cfg, {"generated": generated, "status": "complete"}, option) is False
    assert cc.record_is_current(cfg, {"status": "complete"}, option) is False       # no 'generated' at all


@pytest.mark.parametrize("option", OPTIONS)
def test_record_older_than_the_velocity_is_not_current(tmp_path, option):
    """A record from a previous MintPy run: the velocity was rewritten after it."""
    t = T
    cfg = _cfg(tmp_path, option)
    _velocity(cfg, t)
    assert cc.record_is_current(cfg, _rec(t - 600), option) is False
    assert cc.record_is_current(cfg, _rec(t - 5), option) is False


@pytest.mark.parametrize("option", OPTIONS)
def test_record_at_or_after_the_velocity_is_current(tmp_path, option):
    t = T
    cfg = _cfg(tmp_path, option)
    _velocity(cfg, t)
    assert cc.record_is_current(cfg, _rec(t), option) is True          # same second (written right after)
    assert cc.record_is_current(cfg, _rec(t + 600), option) is True


@pytest.mark.parametrize("option", OPTIONS)
def test_geocoded_velocity_is_the_one_compared(tmp_path, option):
    """With a geocoded velocity, that product's time counts (crosscheck.product),
    not the older radar velocity's."""
    t = T
    cfg = _cfg(tmp_path, option)
    _velocity(cfg, t)                       # radar velocity, older than the record
    _velocity(cfg, t + 1200, geo=True)      # geocoded again after the record
    assert cc.record_is_current(cfg, _rec(t + 600), option) is False
    assert cc.record_is_current(cfg, _rec(t + 1800), option) is True


@pytest.mark.parametrize("option", OPTIONS)
def test_no_velocity_product_is_current(tmp_path, option):
    cfg = _cfg(tmp_path, option)
    assert not cc.product(cfg, "velocity.h5").exists()
    assert cc.record_is_current(cfg, _rec(T), option) is True
