"""HRRR as the weather default for new projects (24 Oct 2026, Kyle: the
Copernicus ERA5 queue had been clogged since 22 Oct). areas.weather_default:
HRRR through RAiDER on the MintPy routes where it serves the whole record
(continental US, from 2016-07-15) and raider-env is installed, ERA5
otherwise; the preflight checks the environment and the model's range; the
installer builds raider-env from install/envs/raider-env.yml."""
import re
from pathlib import Path

import pytest

from insar_pipeline import areas, dashboard as db, settings
import test_fix_L9a as l9

home = l9.home          # the dashboard tests' fixture: a temporary projects folder

PKG = Path(__file__).resolve().parents[1]
INDIANA = (-87.62, 38.20, -87.27, 38.47)
APPIN = (150.70, -34.30, 150.90, -34.10)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """raider-env 'installed' under a temporary conda base."""
    base = tmp_path / "conda"
    (base / "envs" / areas.HRRR_ENV / "bin").mkdir(parents=True)
    (base / "envs" / areas.HRRR_ENV / "bin" / "raider.py").write_text("")
    monkeypatch.setattr(settings, "env_prefix", lambda name: base / "envs" / name)
    return base


def test_hrrr_in_the_us_and_the_start_moves_to_its_first_day(env):
    w, start, note = areas.weather_default("miaplpy", INDIANA, "2016-01-01")
    assert w == {"mintpy_tropo_method": "raider", "weather_model": "HRRR", "raider_env": "raider-env"}
    assert start == "2016-07-15" and "2016-07-15" in note and "asked for 2016-01-01" in note
    w, start, note = areas.weather_default("stack", INDIANA, "2019-03-01")
    assert w["weather_model"] == "HRRR" and start == "2019-03-01" and "asked for" not in note


def test_era5_outside_the_us_on_disp_and_without_the_environment(env, monkeypatch):
    w, start, note = areas.weather_default("miaplpy", APPIN, "2016-01-01")
    assert w == {"mintpy_tropo_method": "pyaps"} and start == "2016-01-01" and "continental US" in note
    assert areas.weather_default("disp", INDIANA, "2016-01-01") == ({"mintpy_tropo_method": "pyaps"}, "2016-01-01", "")
    monkeypatch.setattr(settings, "env_prefix", lambda name: Path("/nonexistent") / name)
    w, start, note = areas.weather_default("miaplpy", INDIANA, "2016-01-01")
    assert w == {"mintpy_tropo_method": "pyaps"} and start == "2016-01-01" and "not installed" in note


def test_a_new_project_carries_hrrr_and_says_so(env, tmp_path, monkeypatch):
    monkeypatch.setattr(areas, "subswaths", lambda *a, **k: [2])
    monkeypatch.setattr(areas, "burst_coverage", lambda area, track, direction, targets=None, **k: {
        "swaths": [2], "bursts": 4, "covered": 1.0, "targets": {n: 1.0 for n in (targets or {})}})
    zone = areas.box(38.29, -87.52, 38.37, -87.42, "Zone")
    out = areas.write_project("HRRR Site", zone, "miaplpy", "2016-01-01", "2026-10-01", track=19, direction="ascending",
                              report_areas=[("Full Satellite Imaging", "full"), ("Zone", zone)],
                              projects_root=tmp_path / "projects", configs_dir=tmp_path / "configs")
    text = Path(out).read_text()
    assert "# Weather delays from NOAA HRRR" in text
    assert "mintpy_tropo_method: raider" in text and "weather_model: HRRR" in text and "raider_env: raider-env" in text
    assert "start_date: '2016-07-15'" in text


def test_preflight_fails_without_raider_env_and_checks_the_model_range(env, home, monkeypatch):
    """On a fake finished project (the dashboard tests' fixture), a MintPy restart's preflight."""
    from insar_pipeline import stack_backend as sb
    from insar_pipeline.config import load_config
    f = l9._project(home)
    cfg = load_config(str(f))
    cfg.mintpy_tropo_method, cfg.weather_model, cfg.raider_env = "raider", "HRRR", "raider-env"
    cfg.start_date = "2016-07-15"
    monkeypatch.setattr(sb, "planned_dates", lambda c, rec=None: None)
    monkeypatch.setattr(sb, "check_weather_credentials", lambda c: pytest.fail("no ERA5 check on the HRRR route"))
    wx = lambda res: [(lv, t) for lv, t in res if t.startswith("weather") or "raider-env" in t]  # noqa: E731
    lines = wx(db.preflight_checks(cfg, str(f), "mintpy"))
    assert lines == [("ok", lines[0][1])] and "HRRR covers this area" in lines[0][1]
    monkeypatch.setattr(sb, "planned_dates", lambda c, rec=None: {"20160327", "20160820"})
    lines = wx(db.preflight_checks(cfg, str(f), "mintpy"))
    assert any(lv == "fail" and "2016-07-15" in t and "mintpy_exclude_dates" in t for lv, t in lines)
    cfg.mintpy_exclude_dates = ["20160327"]                   # left out of the series: no longer a problem
    assert [lv for lv, _ in wx(db.preflight_checks(cfg, str(f), "mintpy"))] == ["ok"]
    monkeypatch.setattr(settings, "env_prefix", lambda name: Path("/nonexistent") / name)
    lines = wx(db.preflight_checks(cfg, str(f), "mintpy"))
    assert any(lv == "fail" and "raider-env" in t and "install.sh" in t for lv, t in lines)


def test_the_installer_builds_raider_env_from_a_pinned_file():
    sh = (PKG / "install" / "install.sh").read_text()
    assert re.search(r"for ENV in [^;]*\braider-env\b", sh)
    yml = (PKG / "install" / "envs" / "raider-env.yml").read_text()
    assert "name: raider-env" in yml and re.search(r"^  - raider=\S+$", yml, re.M)
    assert not re.search(r"^  - _x86_64-microarch", yml, re.M)  # CPU-specific (portable-install trap)
    assert re.search(r"^    - herbie-data==\S+$", yml, re.M)
