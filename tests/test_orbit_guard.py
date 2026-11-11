"""Orbit staging guard: a scene left without an orbit file is retried, then
stops the run with a plain message instead of crashing stackSentinel on a
Copernicus login prompt (study_areav2, 10 Nov 2026: one scene's download came
back empty without raising)."""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import data_acquisition as da  # noqa: E402
from insar_pipeline import stack_backend as sb  # noqa: E402
from insar_pipeline.config import PipelineConfig  # noqa: E402

S1D_SCENE = "S1D_IW_SLC__1SSV_20261031T015818_20261031T015821_005253_009FB1_B29B.SAFE"
S1D_RESORB = "S1D_OPER_AUX_RESORB_OPOD_20261031T054918_V20261031T014849_20261031T052859.EOF"


def scene(d, sat="S1A"):
    return Path(f"{sat}_IW_SLC__1SSV_{d}T015818_{d}T015821_012284_01319D_7C1C.SAFE")


def eof_for(d, sat="S1A"):
    return f"{sat}_OPER_AUX_POEORB_OPOD_X_V{d}T000000_{d}T235959.EOF"


def make_cfg(tmp_path):
    cfg = PipelineConfig(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.52, 38.47],
                         start_date="2020-01-01", end_date="2020-12-31", backend="miaplpy")
    cfg.orbit_dir.mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
def fake_eof(monkeypatch):
    """download_eofs that writes a covering file unless the date is in `empty`
    (returns [] without raising, the failure seen in the field)."""
    state = {"empty": set(), "calls": []}

    def download_eofs(**kw):
        d = Path(kw["sentinel_file"]).name[17:25]
        state["calls"].append(d)
        if d in state["empty"]:
            return []
        f = Path(kw["save_dir"]) / eof_for(d)
        f.touch()
        return [f]

    mod = types.ModuleType("eof.download")
    mod.download_eofs = download_eofs
    monkeypatch.setitem(sys.modules, "eof", types.ModuleType("eof"))
    monkeypatch.setitem(sys.modules, "eof.download", mod)
    monkeypatch.setattr(da, "ORBIT_RETRY_PAUSES_S", (0, 0))
    monkeypatch.setattr(da, "_has_copernicus_netrc", lambda: False)
    return state


def test_coverage_rule_matches_stacksentinel(tmp_path):
    (tmp_path / S1D_RESORB).touch()
    s = Path(S1D_SCENE)
    assert da.orbitless_scenes([s], tmp_path) == []
    # wrong platform: an S1A file does not cover an S1D scene
    (tmp_path / S1D_RESORB).rename(tmp_path / S1D_RESORB.replace("S1D", "S1A", 1))
    assert da.orbitless_scenes([s], tmp_path) == [s]
    # window ends inside the 60 s margin of the scene: not covered
    (tmp_path / "S1D_OPER_AUX_RESORB_OPOD_X_V20261031T014849_20261031T015900.EOF").touch()
    assert da.orbitless_scenes([s], tmp_path) == [s]
    # unparseable scene names are left to stackSentinel
    assert da.orbitless_scenes([Path("odd.SAFE")], tmp_path) == []


def test_empty_download_is_retried_and_recovers(tmp_path, fake_eof):
    cfg = make_cfg(tmp_path)
    fake_eof["empty"] = {"20261031"}
    orig = sys.modules["eof.download"].download_eofs

    def flaky(**kw):          # empty on the first call only
        if fake_eof["calls"].count("20261031") >= 1:
            fake_eof["empty"].discard("20261031")
        return orig(**kw)
    sys.modules["eof.download"].download_eofs = flaky
    da.download_orbits(cfg, [scene("20261025"), scene("20261031")], "u", "p")
    assert fake_eof["calls"].count("20261031") == 2
    assert da.orbitless_scenes([scene("20261031")], cfg.orbit_dir) == []


def test_still_missing_raises_plain_error(tmp_path, fake_eof):
    cfg = make_cfg(tmp_path)
    fake_eof["empty"] = {"20261031"}
    with pytest.raises(da.OrbitsMissing, match=r"1 of 2 scene\(s\) \(20261031\).*Resume"):
        da.download_orbits(cfg, [scene("20261025"), scene("20261031")], "u", "p")
    assert fake_eof["calls"].count("20261031") == 3          # first try + 2 retries


def test_missing_with_copernicus_login_is_left_to_stacksentinel(tmp_path, fake_eof, monkeypatch):
    cfg = make_cfg(tmp_path)
    fake_eof["empty"] = {"20261031"}
    monkeypatch.setattr(da, "_has_copernicus_netrc", lambda: True)
    da.download_orbits(cfg, [scene("20261031")], "u", "p")   # no raise


def test_orbit_already_on_disk_counts(tmp_path, fake_eof):
    """An orbit staged earlier (e.g. by hand) covers the scene even when this
    run's download comes back empty."""
    cfg = make_cfg(tmp_path)
    (cfg.orbit_dir / eof_for("20261031")).touch()
    fake_eof["empty"] = {"20261031"}
    da.download_orbits(cfg, [scene("20261031")], "u", "p")
    assert fake_eof["calls"] == ["20261031"]


def test_stage_inputs_lets_orbits_missing_through(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(PipelineConfig, "ensure_dirs", lambda self: None)
    monkeypatch.setattr(da, "download_orbits",
                        lambda *a, **k: (_ for _ in ()).throw(da.OrbitsMissing("x")))
    dem = []
    monkeypatch.setattr(da, "get_dem", lambda c: dem.append(1))
    with pytest.raises(da.OrbitsMissing):
        sb.stage_inputs(cfg, "u", "p", skip_download=True)
    assert not dem                                           # stopped before the DEM
    # any other orbit failure keeps the old behaviour: warn and carry on
    monkeypatch.setattr(da, "download_orbits",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("y")))
    sb.stage_inputs(cfg, "u", "p", skip_download=True)
    assert dem == [1]
