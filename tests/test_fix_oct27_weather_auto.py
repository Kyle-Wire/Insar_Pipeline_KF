"""
27 Oct 2026: weather dates are handled by the run, never by a config edit.

Site E descending stopped after its inversion: "HRRR weather delays are
missing for 3 of 292 dates (20180305, 20211027, 20240226)". The advice was to
add them to mintpy_exclude_dates (the user added them to allow_missing_dates,
which is about radar images, and the run failed again). None of the three was
really unavailable: runs stopped part-way had left cut-off HRRR downloads in
weather_files/, which Herbie reused on every attempt; a fresh download gave a
cube for each.

- tropo_disp.delay_cube_for_date tries a failed date once more from a clean
  download cache;
- weather_raider.check_dates fetches every stack date's weather before the
  inversion and records the dates still without any (with the reason) in
  tropo_corrections/<MODEL>/unavailable_dates.json, rewritten by every check;
- time_series.exclude_weather_unavailable leaves those dates out of the series
  (template excludeDate, modify_network re-run) when the set changed.
"""
import json
import sys
import types
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import procs, tropo_disp                    # noqa: E402
from insar_pipeline import time_series as ts                     # noqa: E402
from insar_pipeline import weather_raider as wr                  # noqa: E402


def _cfg(tmp_path, method="raider"):
    P = tmp_path / "proj"
    (P / "mintpy").mkdir(parents=True, exist_ok=True)
    return types.SimpleNamespace(project_dir=P, mintpy_dir=P / "mintpy", tropo_dir=P / "tropo_corrections",
                                 weather_model="HRRR", raider_env="", mintpy_tropo_method=method,
                                 mintpy_exclude_dates=["20160322"], bbox=[-100.0, 40.0, -99.9, 40.1])


def test_a_failed_date_is_tried_again_from_a_clean_download(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls = []

    def run_group(cmd, cwd):
        cwd = Path(cwd)
        calls.append((cwd / "weather_files").exists())
        if len(calls) == 1:                                  # the cut-off download left by a stopped run
            return types.SimpleNamespace(returncode=1, stdout="ERROR: cannot rename 'gh'\n", stderr="")
        (cwd / "HRRR_tropo_20211027T130201_ztd.nc").write_bytes(b"")
        return types.SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
    monkeypatch.setattr(procs, "run_group", run_group)
    out = cfg.tropo_dir / "HRRR" / "20211027"
    (out / "weather_files" / "hrrr").mkdir(parents=True)
    (out / "weather_files" / "hrrr" / "subset_x.grib2").write_bytes(b"cut off")
    got = tropo_disp.delay_cube_for_date(cfg, datetime(2021, 10, 27, 13, 2, 1))
    assert got is not None and got.name.endswith("_ztd.nc")
    assert calls == [True, False]                             # the second try had no cache to reuse
    log = (out / "raider_run.log").read_text()
    assert "cannot rename 'gh'" in log and "retry with the download cache cleared" in log


def test_the_record_registers_and_takes_back_dates(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cubes = {"20180305": None, "20211027": tmp_path / "c.nc", "20240226": None, "20160322": None}
    monkeypatch.setattr(wr, "fetch_cubes", lambda c, dates, workers=None: {d: cubes[d] for d in dates})
    gap = cfg.tropo_dir / "HRRR" / "20180305"
    gap.mkdir(parents=True)
    (gap / "raider_run.log").write_text("Did not find model=hrrr 2018-Mar-05 14:00 UTC F00\n")
    got = wr.check_dates(cfg, ["20180305", "20211027", "20240226", "20160322"])
    assert set(got) == {"20180305", "20240226", "20160322"}
    assert "archive has no file" in got["20180305"] and "clean download" in got["20240226"]
    assert "before HRRR's first day" in got["20160322"]
    assert wr.registered_unavailable(cfg) == got
    rec = json.loads(wr.unavailable_path(cfg).read_text())
    assert rec["model"] == "HRRR" and rec["checked"] == 4
    cubes["20240226"] = tmp_path / "d.nc"                     # available now: taken back
    assert "20240226" not in wr.check_dates(cfg, ["20180305", "20240226"])
    assert set(wr.registered_unavailable(cfg)) == {"20180305"}
    assert wr.registered_unavailable(_cfg(tmp_path / "e", method="pyaps")) == {}


def test_the_series_leaves_them_out_before_the_inversion(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    tmpl = cfg.mintpy_dir / "smallbaselineApp.cfg"
    tmpl.write_text("mintpy.network.excludeDate = 20160322\n")
    stack = ["20160322", "20180305", "20211027", "20240226"]
    monkeypatch.setattr(ts, "kept_network", lambda c: (stack, [], []))
    asked, runs = [], []

    def check(c, dates, workers=None):
        asked.append(list(dates))
        reasons = {"20180305": "NOAA's HRRR archive has no file for the hours around the pass"}
        wr.unavailable_path(c).parent.mkdir(parents=True, exist_ok=True)
        wr.unavailable_path(c).write_text(json.dumps({"dates": reasons}))
        return reasons
    monkeypatch.setattr(wr, "check_dates", check)
    monkeypatch.setattr(ts, "_mintpy", lambda c, p, a, b, log: runs.append((a, b)) or types.SimpleNamespace(returncode=0))
    assert ts.exclude_weather_unavailable(cfg, tmpl) == ["20180305"]
    assert asked == [["20180305", "20211027", "20240226"]]   # the config's own exclusions are not fetched
    assert "mintpy.network.excludeDate = 20160322,20180305" in tmpl.read_text()
    assert runs == [("modify_network", "modify_network")]
    assert ts.series_exclude_dates(cfg) == ["20160322", "20180305"]
    # the same set again (a resume): nothing rewritten, modify_network not re-run
    assert ts.exclude_weather_unavailable(cfg, tmpl) == ["20180305"] and len(runs) == 1
    # off the RAiDER route: no check at all
    cfg.mintpy_tropo_method = "pyaps"
    assert ts.exclude_weather_unavailable(cfg, tmpl) == [] and len(asked) == 2


# ------------------------------------------------ 30 Oct 2026: delays that are not physical
def _write_cube(path, hydro_sea=2.3, wet_sea=0.15):
    """A RAiDER-shaped cube: delays falling off with height over a 3 x 3 grid."""
    import numpy as np
    import xarray as xr
    z = np.array([-100.0, 0.0, 1000.0, 5000.0, 15000.0])
    f = np.exp(-np.clip(z, 0, None) / 8000.0)[:, None, None] * np.ones((1, 3, 3))
    ds = xr.Dataset({"hydro": (("z", "y", "x"), hydro_sea * f), "wet": (("z", "y", "x"), wet_sea * f ** 4)},
                    coords={"z": z, "y": [40.1, 40.05, 40.0], "x": [-100.0, -99.95, -99.9]})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return Path(path)


def test_a_cube_that_is_not_physical_is_refused(tmp_path):
    """The first city site's 2018-08-04 HRRR cube: -5,787 m hydrostatic, the
    wet delay negative, RAiDER exiting cleanly; subtracted, a 58 m step."""
    import numpy as np
    assert tropo_disp.cube_problem(_write_cube(tmp_path / "ok.nc")) == ""
    bad = _write_cube(tmp_path / "bad.nc", hydro_sea=-5787.0, wet_sea=-0.46)
    assert "hydrostatic delay" in tropo_disp.cube_problem(bad)
    assert "wet delay" in tropo_disp.cube_problem(_write_cube(tmp_path / "wet.nc", wet_sea=3.0))
    one = np.zeros((2, 2))
    assert tropo_disp._sample_cube(bad, one - 99.95, one + 40.05, one) is None
    z = tropo_disp._sample_cube(tmp_path / "ok.nc", one - 99.95, one + 40.05, one)
    assert z is not None and np.allclose(z, 2.3 + 0.15, atol=1e-3)


def test_such_a_date_is_fetched_again_then_left_out_with_the_reason(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    root = cfg.tropo_dir / "HRRR"
    good = _write_cube(root / "20180723" / "HRRR_tropo_20180723T225920_ztd.nc")
    bad = _write_cube(root / "20180804" / "HRRR_tropo_20180804T225920_ztd.nc", hydro_sea=-5787.0, wet_sea=-0.46)
    fixable = _write_cube(root / "20180816" / "HRRR_tropo_20180816T225920_ztd.nc", hydro_sea=-5787.0)
    monkeypatch.setattr(wr, "fetch_cubes", lambda c, dates, workers=None: {
        "20180723": good, "20180804": bad, "20180816": fixable})
    monkeypatch.setattr(wr, "acquisition_time", lambda c: "22:59:20")
    again = []

    def one_cube(c, d, t):                                    # the clean download: same garbage, or a good cube
        again.append(d)
        return _write_cube(root / d / f"HRRR_tropo_{d}T225920_ztd.nc",
                           hydro_sea=-5787.0 if d == "20180804" else 2.3)
    monkeypatch.setattr(wr, "_one_cube", one_cube)
    got = wr.check_dates(cfg, ["20180723", "20180804", "20180816"])
    assert sorted(again) == ["20180804", "20180816"]          # the physical cube is not fetched again
    assert list(got) == ["20180804"] and "not physical" in got["20180804"] and "clean download" in got["20180804"]
    assert (root / "20180804" / (bad.name + ".implausible")).exists() and not bad.exists()
    assert fixable.exists() and tropo_disp.cube_problem(fixable) == ""
    assert wr.registered_unavailable(cfg) == got              # so the series leaves it out before the inversion
