"""
24 Oct 2026: weather delays from NOAA HRRR on the MintPy routes, and an
empty ERA5.h5 that broke every re-run.

- MintPy applies any inputs/<MODEL>.h5 on the series' grid, whatever dates
  it holds. One written while ERA5 was unreachable held none, and every
  re-run died in diff.py ("IndexError: list index out of range"). The run
  now moves such a file aside before the troposphere step
  (time_series.retire_partial_weather_file).
- mintpy_tropo_method: raider (weather_model HRRR) writes inputs/HRRR.h5 from
  RAiDER cubes (weather_raider.py) in pyaps3's layout and sign, and tells
  MintPy method pyaps / weatherModel HRRR, so MintPy's own diff.py applies it.
  Checked here against MintPy's diff.py itself.
- The HRRR name is a weather token everywhere a series' name is read
  (deliverables, report wording, provenance).

Synthetic projects in tmp_path; no network, no RAiDER (cubes are put in its
cache, which delay_cube_for_date reuses).
"""
import json
from datetime import datetime
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import corrections                          # noqa: E402
from insar_pipeline import time_series as ts_mod                 # noqa: E402
from insar_pipeline import weather_raider as wr                  # noqa: E402

DATES = ["20160322", "20170105", "20170117", "20170129"]
NY, NX = 6, 7


def _write_ts(path: Path, dates, data=None, ref=(2, 3)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["date"] = np.array([d.encode() for d in dates])
        f["timeseries"] = (np.zeros((len(dates), NY, NX), "float32") if data is None else data.astype("float32"))
        f.attrs.update({"FILE_TYPE": "timeseries", "LENGTH": str(NY), "WIDTH": str(NX), "UNIT": "m",
                        "REF_Y": str(ref[0]), "REF_X": str(ref[1]), "REF_DATE": dates[0] if dates else "",
                        "REF_LAT": "38.9", "REF_LON": "-104.8"})


def _write_geom(path: Path):
    lat = np.linspace(38.95, 38.88, NY)[:, None] * np.ones((1, NX))
    lon = np.ones((NY, 1)) * np.linspace(-104.89, -104.78, NX)[None, :]
    height = 1800 + 300 * np.random.default_rng(0).random((NY, NX))
    inc = np.full((NY, NX), 38.0)
    inc[0, 0] = 0.0                                          # no geometry: pyaps3 leaves 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in (("latitude", lat), ("longitude", lon), ("height", height), ("incidenceAngle", inc)):
            f[k] = v.astype("float32")
        f.attrs.update({"FILE_TYPE": "geometry", "LENGTH": str(NY), "WIDTH": str(NX)})
    return lat, lon, height, inc


def _zenith(z, y, x, k):
    """The synthetic total zenith delay (m) of date k: falls with height,
    tilts with latitude, changes date to date."""
    return 2.4 - 3.1e-4 * z + 0.01 * (y - 38.9) * (k + 1) + 0.002 * k + 0.0 * x


def _write_cube(tropo_dir: Path, date: str, k: int):
    import xarray as xr
    d = tropo_dir / "HRRR" / date
    d.mkdir(parents=True, exist_ok=True)
    z = np.array([-500.0, 0, 1000, 2000, 3000, 5000])
    y = np.linspace(39.2, 38.6, 13)
    x = np.linspace(-105.1, -104.5, 13)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    total = _zenith(Z, Y, X, k)
    xr.Dataset({"wet": (("z", "y", "x"), 0.2 * total), "hydro": (("z", "y", "x"), 0.8 * total)},
               coords={"z": z, "y": y, "x": x}).to_netcdf(d / f"HRRR_tropo_{date}T005345_ztd.nc")


@pytest.fixture
def proj(tmp_path, monkeypatch):
    P = tmp_path / "proj"
    cfg = types.SimpleNamespace(project_dir=P, mintpy_dir=P / "mintpy", tropo_dir=P / "tropo_corrections",
                                weather_model="HRRR", raider_env="", mintpy_tropo_method="raider",
                                bbox=[-104.89, 38.87, -104.78, 38.96], era5_allow_partial=False)
    P.mkdir(parents=True)
    monkeypatch.setattr(wr, "acquisition_time", lambda c: "00:53:45")
    return cfg


# ------------------------------------------------------------------ empty / partial delay file

def test_an_empty_or_partial_weather_file_is_set_aside_a_complete_one_kept(proj):
    M = proj.mintpy_dir
    _write_ts(M / "timeseries.h5", DATES)
    w = M / "inputs" / "ERA5.h5"
    _write_ts(w, [])                                         # what the 23 Oct run left: no dates
    moved = ts_mod.retire_partial_weather_file(proj, w, "20261024_120000")
    assert not w.exists() and moved.exists() and "superseded/run_20261024_120000/inputs" in str(moved)
    _write_ts(w, DATES[:-1])
    assert ts_mod.retire_partial_weather_file(proj, w, "20261024_130000") is not None and not w.exists()
    _write_ts(w, DATES)
    assert ts_mod.retire_partial_weather_file(proj, w, "20261024_140000") is None and w.exists()


# ------------------------------------------------------------------ MintPy settings

def test_raider_reaches_mintpy_as_pyaps_with_the_model_named():
    raider = types.SimpleNamespace(mintpy_tropo_method="raider", weather_model="hrrr")
    era5 = types.SimpleNamespace(mintpy_tropo_method="pyaps", weather_model="ERA5")
    assert corrections.mintpy_tropo_method(raider) == "pyaps" and corrections.weather_model(raider) == "HRRR"
    assert corrections.mintpy_tropo_method(era5) == "pyaps" and corrections.weather_model(era5) == "ERA5"
    lines = corrections.mintpy_tropo_lines(raider)
    assert lines[0].endswith("= pyaps") and lines[1].endswith("= HRRR")
    # every template writer goes through the helper: MintPy skips an unknown method word silently
    import inspect
    from insar_pipeline import miaplpy_backend, stack_backend
    for fn in (stack_backend.mintpy_loader_lines, miaplpy_backend.write_template, ts_mod.run_mintpy_with_loader):
        src = inspect.getsource(fn)
        assert "mintpy_tropo_lines(cfg" in src and "troposphericDelay.method       = {cfg" not in src


def test_config_accepts_raider_with_a_raider_model(tmp_path):
    from insar_pipeline.config import ConfigError, load_config
    base = (f"project_dir: {tmp_path / 'p'}\nbbox: [-104.9, 38.87, -104.78, 38.96]\n"
            "start_date: '2017-01-01'\nend_date: '2018-01-01'\nmintpy_tropo_method: raider\n")
    f = tmp_path / "ok.yaml"
    f.write_text(base + "weather_model: HRRR\n")
    assert load_config(str(f)).mintpy_tropo_method == "raider"
    f.write_text(base + "weather_model: NARR\n")
    with pytest.raises(ConfigError, match="weather_model"):
        load_config(str(f))


# ------------------------------------------------------------------ the delay file

def test_hrrr_file_has_pyaps3_layout_and_sign_and_mintpy_applies_it(proj):
    """inputs/HRRR.h5 = -(zenith at each pixel's height / cos(incidence));
    MintPy's own diff.py then removes the delay referenced to the series'
    reference pixel and date."""
    M = proj.mintpy_dir
    rng = np.random.default_rng(1)
    ts_data = rng.normal(0, 0.003, (len(DATES), NY, NX))
    ts_data[0] = 0.0
    _write_ts(M / "timeseries.h5", DATES, ts_data)
    with h5py.File(M / "timeseries.h5", "a") as f:
        f.attrs["REF_DATE"] = DATES[1]
    lat, lon, height, inc = _write_geom(M / "inputs" / "geometryRadar.h5")
    for k, d in enumerate(DATES):
        _write_cube(proj.tropo_dir, d, k)
    out = M / "inputs" / "HRRR.h5"
    rec = wr.build_delay_file(proj, M / "timeseries.h5", M / "inputs" / "geometryRadar.h5", out)
    assert rec["dates_written"] == 3 and rec["missing_dates"] == ["20160322"]     # before HRRR starts
    with h5py.File(out, "r") as f:
        got_dates = [d.decode() for d in f["date"][:]]
        delay = f["timeseries"][:]
        attrs = dict(f.attrs)
    assert got_dates == DATES[1:]
    assert attrs["FILE_TYPE"] == "timeseries" and attrs["UNIT"] == "m"
    assert not {"REF_Y", "REF_X", "REF_DATE", "REF_LAT", "REF_LON"} & set(attrs)
    for i, d in enumerate(got_dates):
        k = DATES.index(d)
        expect = -_zenith(height, lat, lon, k) / np.cos(np.radians(inc))
        good = inc != 0
        assert np.allclose(delay[i][good], expect[good], atol=2e-4)
        assert delay[i][0, 0] == 0.0
    assert json.loads((M / wr.RECORD).read_text())["model"] == "HRRR"

    # through MintPy's diff.py, on the dates both files hold
    mdiff = pytest.importorskip("mintpy.diff")
    _write_ts(M / "ts3.h5", DATES[1:], ts_data[1:])
    with h5py.File(M / "ts3.h5", "a") as f:
        f.attrs["REF_DATE"] = DATES[1]
    mdiff.diff_timeseries(str(M / "ts3.h5"), str(out), str(M / "corrected.h5"), force_diff=True)
    with h5py.File(M / "corrected.h5", "r") as f:
        corrected = f["timeseries"][:]
    dref = delay - delay[:, 2:3, 3:4]                         # reference pixel (2, 3)
    dref = dref - dref[0]                                      # reference date
    assert np.allclose(corrected, ts_data[1:] - dref, atol=1e-6)


def test_a_date_without_a_cube_stops_the_run_unless_partial_is_allowed(proj, monkeypatch):
    M = proj.mintpy_dir
    _write_ts(M / "timeseries.h5", DATES[1:])
    _write_geom(M / "inputs" / "geometryRadar.h5")
    for k, d in enumerate(DATES[1:3]):                        # the last date has no cube
        _write_cube(proj.tropo_dir, d, k)
    calls = []

    def cached_only(cfg, date, acq_time):                     # RAiDER failing for the last date
        calls.append(date)
        got = sorted((cfg.tropo_dir / "HRRR" / date).glob("*.nc"))
        return got[0] if got else None
    monkeypatch.setattr(wr, "_one_cube", cached_only)
    rec = wr.build_delay_file(proj, M / "timeseries.h5", M / "inputs" / "geometryRadar.h5",
                              M / "inputs" / "HRRR.h5")
    assert rec["missing_dates"] == [DATES[3]] and len(calls) == 3
    with pytest.raises(RuntimeError, match="HRRR"):
        corrections.verify_era5_coverage(proj, M / "timeseries.h5", M / "inputs" / "HRRR.h5", "HRRR")
    proj.era5_allow_partial = True
    assert corrections.verify_era5_coverage(proj, M / "timeseries.h5", M / "inputs" / "HRRR.h5",
                                            "HRRR")["dates_corrected"] == 2


def test_a_date_missing_from_the_archive_is_named_with_the_remedy(proj):
    """Site B 2016-08-08 (26 Oct 2026): NOAA's archive has no HRRR file for
    the pass, so 'wait and re-run' (the ERA5 advice it used to give) cannot
    work; the message says so and names mintpy_exclude_dates."""
    from insar_pipeline import advice
    M = proj.mintpy_dir
    _write_ts(M / "timeseries.h5", DATES[1:])
    _write_ts(M / "inputs" / "HRRR.h5", DATES[1:2])
    gap, flaky = DATES[2], DATES[3]
    for d, text in ((gap, "💔 Did not find ┊ model=hrrr ┊ product=nat ┊ 2016-Aug-08 23:00 UTC F00\n"),
                    (flaky, "requests.exceptions.ConnectionError: Max retries exceeded\n")):
        (proj.tropo_dir / "HRRR" / d).mkdir(parents=True, exist_ok=True)
        (proj.tropo_dir / "HRRR" / d / "raider_run.log").write_text(text)
    assert wr.archive_gaps(proj, [gap, flaky, "20990101"]) == [gap]
    with pytest.raises(corrections.WeatherCoverageError) as e:
        corrections.verify_era5_coverage(proj, M / "timeseries.h5", M / "inputs" / "HRRR.h5", "HRRR")
    msg = str(e.value)
    assert "archive has no file for " + gap in msg and "leaves such dates out of the series by itself" in msg
    assert "mintpy_exclude_dates" not in msg                      # 27 Oct 2026: never an edit for the user
    assert f"RAiDER failed for {flaky}" in msg and "era5_allow_partial" in msg
    assert "Copernicus" not in msg and "5 days" not in msg
    adv = advice.explain(e.value, "processing", cfg=proj)
    assert adv["category"] == "weather" and adv["plain_cause"] == e.value.cause
    assert adv["next_action"].startswith("Resume")

    (proj.tropo_dir / "HRRR" / gap / "raider_run.log").write_text("Timeout\n")     # no gap: retry is the advice
    with pytest.raises(corrections.WeatherCoverageError, match="retries them from a clean download"):
        corrections.verify_era5_coverage(proj, M / "timeseries.h5", M / "inputs" / "HRRR.h5", "HRRR")


def test_the_nisar_route_takes_the_pass_time_from_its_crops(tmp_path):
    """26 Oct 2026 (Site E NISAR): the HRRR step read the pass time from
    Sentinel-1 SAFE names only, so on the NISAR route it stopped with 'no SAFE
    names'. The crops keep each granule's identification; the times below are
    T019A's 2025-10-30 / 2025-11-11 granule over Site E."""
    from insar_pipeline import nisar_backend as nb, tropo_disp
    P = tmp_path / "np_nisar"
    crops = P / "nisar_gunw" / "crops"
    crops.mkdir(parents=True)
    pairs = [("20251030", "2025-10-30T12:11:15.000000000", "2025-10-30T12:11:43.999342105",
              "20251111", "2025-11-11T12:11:16.000000000", "2025-11-11T12:11:44.999342105"),
             ("20251111", "2025-11-11T12:11:16.000000000", "2025-11-11T12:11:44.999342105",
              "20251123", "2025-11-23T12:11:17.000000000", "2025-11-23T12:11:45.999342105")]
    for r, r0, r1, s, s0, s1 in pairs:
        with h5py.File(crops / f"NISAR_GUNW_CROP_T019A_{r}_{s}.h5", "w") as f:
            for k, v in (("referenceZeroDopplerStartTime", r0), ("referenceZeroDopplerEndTime", r1),
                         ("secondaryZeroDopplerStartTime", s0), ("secondaryZeroDopplerEndTime", s1)):
                f[nb.IDENT + k] = v.encode()
    with h5py.File(crops / "NISAR_GUNW_CROP_T019A_20251030_20251111_alt.h5", "w") as f:   # a side file: ignored
        f[nb.IDENT + "referenceZeroDopplerStartTime"] = b"2025-10-30T03:00:00"
    got = nb.acquisition_times(types.SimpleNamespace(project_dir=P))
    assert sorted(got) == ["20251030", "20251111", "20251123"]
    assert got["20251030"] == 12 * 3600 + 11 * 60 + 29          # whole seconds (_zd_time)
    cfg = types.SimpleNamespace(project_dir=P, backend="nisar", weather_model="HRRR",
                                bbox=[-100.0, 40.0, -99.9, 40.1])
    assert wr.acquisition_time(cfg) == "12:11:30"                # median of :29, :30, :31
    assert "look_dir: left" in tropo_disp._config_text(cfg, datetime(2025, 10, 30, 12, 11, 30), tmp_path)
    cfg.backend = "miaplpy"
    assert "look_dir: right" in tropo_disp._config_text(cfg, datetime(2025, 10, 30, 12, 11, 30), tmp_path)
    with pytest.raises(RuntimeError, match="NISAR crops"):
        wr.acquisition_time(types.SimpleNamespace(project_dir=tmp_path / "empty", backend="nisar"))


def test_a_complete_file_is_reused_without_fetching(proj, monkeypatch):
    M = proj.mintpy_dir
    _write_ts(M / "timeseries.h5", DATES[1:])
    _write_ts(M / "inputs" / "HRRR.h5", DATES[1:])
    monkeypatch.setattr(wr, "fetch_cubes", lambda *a, **k: pytest.fail("must not fetch"))
    assert wr.build_delay_file(proj, M / "timeseries.h5", M / "inputs" / "geometryRadar.h5",
                               M / "inputs" / "HRRR.h5")["reused"] is True


# ------------------------------------------------------------------ the name downstream

def test_the_hrrr_series_is_the_corrected_one_and_is_described_as_hrrr(tmp_path):
    from insar_pipeline import deliverables as dl, provenance, report_insights as ri
    cfg = types.SimpleNamespace(mintpy_dir=tmp_path / "mintpy", backend="miaplpy")
    for n in ("timeseries.h5", "timeseries_ramp_demErr.h5", "timeseries_HRRR_ramp_demErr.h5"):
        _write_ts(cfg.mintpy_dir / n, DATES[1:])
    assert dl._timeseries_by_tokens(cfg).name == "timeseries_HRRR_ramp_demErr.h5"
    words = ri.corrections_applied(cfg, cfg.mintpy_dir / "timeseries_HRRR_ramp_demErr.h5")
    assert "weather delays (NOAA HRRR weather model)" in words
    assert provenance._weather_label(["HRRR", "ramp", "demErr"]) == "RAiDER (HRRR)"
    assert provenance._weather_label(["ERA5", "demErr"]) == "pyaps (ERA5)"
    assert provenance._weather_label(["ramp"]) == "none"


def test_a_reference_date_before_hrrr_is_refused_in_words(proj):
    """The project that met it: its first date (the series' reference date)
    is 22 Mar 2016, four months before HRRR's archive starts."""
    M = proj.mintpy_dir
    _write_ts(M / "timeseries.h5", DATES)                     # REF_DATE = 20160322
    _write_geom(M / "inputs" / "geometryRadar.h5")
    for k, d in enumerate(DATES):
        _write_cube(proj.tropo_dir, d, k)
    with pytest.raises(RuntimeError, match="reference date 20160322 has no HRRR delay .*before 2016-07-15"):
        wr.build_delay_file(proj, M / "timeseries.h5", M / "inputs" / "geometryRadar.h5", M / "inputs" / "HRRR.h5")


# ------------------------------------------------------------------ dates left out by the config

def test_config_leaves_dates_out_without_calling_them_orphans(monkeypatch, tmp_path):
    """mintpy_exclude_dates (the 5 dates before HRRR starts): every pair
    touching them is dropped by design, so the network check must not take
    them for orphans; a real orphan found later is added to them, not put in
    their place (MintPy keeps only the last excludeDate line)."""
    cfg = types.SimpleNamespace(mintpy_dir=tmp_path, max_temp_baseline=48, max_perp_baseline=200.0,
                                mintpy_exclude_dates=[20160322, "20160427"])
    cfgp = tmp_path / "smallbaselineApp.cfg"
    cfgp.write_text("mintpy.network.excludeDate = 20160322,20160427\n")
    dates = ["20160322", "20160427", "20160801", "20160825", "20160918", "20161006"]
    formed = [("20160322_20160427", False, 36, 1.0), ("20160427_20160801", False, 96, 1.0),
              ("20160801_20160825", True, 24, 1.0), ("20160825_20160918", True, 24, 1.0),
              ("20160918_20161006", False, 18, 1.0)]        # 20161006: a real orphan
    calls = []
    monkeypatch.setattr(ts_mod, "kept_network", lambda c: (dates, [p for p, k, _, _ in formed if k], formed))
    monkeypatch.setattr(ts_mod, "_mintpy", lambda *a, **k: calls.append(a[2]) or types.SimpleNamespace(returncode=0))
    info = ts_mod.ensure_connected_network(cfg, cfgp)
    assert info["excluded_dates"] == ["20161006"] and info["excluded_by_config"] == ["20160322", "20160427"]
    assert info["components"] == [3] and calls == ["modify_network"]
    lines = [l for l in cfgp.read_text().splitlines() if l.startswith("mintpy.network.excludeDate ")]
    assert lines == ["mintpy.network.excludeDate = 20160322,20160427,20161006"]
    # nothing else orphaned: the config's dates alone are not a reason to rebuild
    formed[-1] = ("20160918_20161006", True, 18, 1.0)
    calls.clear()
    info = ts_mod.ensure_connected_network(cfg, cfgp)
    assert calls == [] and info["orphaned_dates"] == [] and info["components"] == [4]


def test_mintpy_exclude_dates_is_checked_and_written(tmp_path):
    from insar_pipeline.config import ConfigError, load_config
    base = (f"project_dir: {tmp_path / 'p'}\nbbox: [-104.9, 38.87, -104.78, 38.96]\n"
            "start_date: '2016-01-01'\nend_date: '2018-01-01'\n")
    f = tmp_path / "c.yaml"
    f.write_text(base + "mintpy_exclude_dates: [20160322, '20160427']\n")
    cfg = load_config(str(f))
    assert ts_mod.config_exclude_dates(cfg) == ["20160322", "20160427"]
    f.write_text(base + "mintpy_exclude_dates: ['2016-03-22']\n")
    with pytest.raises(ConfigError, match="YYYYMMDD"):
        load_config(str(f))
    import inspect
    src = inspect.getsource(ts_mod.run_mintpy_with_loader)
    assert "mintpy.network.excludeDate             = {','.join(drop_dates)}" in src
