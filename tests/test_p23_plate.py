"""
Phase 2/3, group plate (Decision 5, 8 Oct 2026): GNSS-referenced products are
expressed relative to the local tectonic plate.

  - plate_motion: the published ITRF2020 plate motion model (Altamimi et al.
    2023) reproduces the model's own published residuals; 'auto' picks the
    plate from where the stations are (and their velocities where two plates
    meet).
  - ground_truth: load_gnss_series(frame='plate') / to_plate_frame remove the
    plate's rigid horizontal motion and nothing else; the default frame is
    still IGS20 (the harness is unchanged).
  - significance: reference_gnss_frame ('plate' default | 'itrf') and
    reference_gnss_plate ('auto' | code), checked.
  - deliverables: the gnss_absolute datum is plate-fixed by default and says
    so; 'itrf' gives the old numbers.
  - real data (read-only, cached NGL series): CA07 (Appin, Australia) and LAMC
    (Indiana) read near zero horizontally after removal.

None of these functions or settings existed before, so every test here fails
on the code in backups/pre_phase23_20261008/.
"""
import hashlib
import json
import logging
import shutil
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from insar_pipeline import deliverables as dl
from insar_pipeline import ground_truth as gt
from insar_pipeline import plate_motion as pm
from insar_pipeline import significance as sg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L7a as L7a                                       # noqa: E402  (synthetic deliverables project)
import test_p1_absref as AB                                      # noqa: E402  (its fake GNSS loader)
import test_p1_gt as G                                           # noqa: E402  (synthetic NGL tenv3 files)

PKG = Path(__file__).resolve().parents[1]
PLATE_DIR = PKG / "review_2026-10-08" / "p23" / "plate"
P1_CACHE = PKG / "review_2026-10-08" / "p1" / "cache"

# ITRF2020-PMM-residuals.dat (Table S2), as published: code, plate, lon (0-360), lat, Ve, Vn, Re, Rn (mm/yr)
PUBLISHED = [
    ("KARR", "AUST", 117.097, -20.981, 38.87, 58.24, -0.029, 0.022),
    ("ALIC", "AUST", 133.886, -23.670, 32.04, 58.97, 0.153, -0.020),
    ("SYDN", "AUST", 151.150, -33.781, 18.31, 54.34, -0.103, -0.160),
    ("PTKL", "AUST", 150.914, -34.476, 17.95, 54.65, -0.100, 0.068),       # Port Kembla, 25 km from Appin
    ("JOZE", "EURA", 21.032, 52.097, 20.95, 14.12, -0.030, 0.033),
    ("LCDT", "NOAM", 272.040, 42.295, -15.66, 0.27, 0.000, 0.157),
    ("ADRI", "NOAM", 275.976, 41.919, -15.73, 1.48, -0.175, -0.029),
    ("HLFX", "NOAM", 296.389, 44.684, -15.21, 8.68, -0.209, 0.164),
    ("TXGA", "NOAM", 265.227, 29.328, -11.85, -2.41, 0.377, -0.133),       # Galveston, near Houston
    ("AMC2", "NOAM", 255.475, 38.803, -14.52, -5.81, -0.226, -0.143),
    ("HILO", "PCFC", 204.947, 19.719, -62.59, 35.38, -0.362, 0.640),
    ("MAUI", "PCFC", 203.743, 20.707, -62.05, 34.59, 0.043, -0.208),
    ("CHAT", "PCFC", 183.434, -43.956, -40.66, 32.94, 0.397, 0.473),
]
APPIN = (-34.2216, 150.6391)            # CA07
SITE_A = (38.4062, -87.7641)        # LAMC
HOUSTON = (29.78, -95.75)
CORCORAN = (36.3234, -119.6228)         # CAHA
POLAND = (50.09, 18.42)                 # Rydultowy


# ===================================================================== plate_motion
def test_pole_table_is_the_published_itrf2020_pmm():
    # the values that matter here, spelled out (deg/Myr, Table S1); 1 deg/Myr = 3.6 mas/yr, so AUST's
    # omega_x 0.4132 is the paper's 1.487 mas/yr
    assert pm.ITRF2020_PMM["AUST"] == (0.4132, 0.3265, 0.3398)
    assert pm.ITRF2020_PMM["NOAM"] == (0.0126, -0.1849, -0.0272)
    assert pm.ITRF2020_PMM["EURA"] == (-0.0237, -0.1442, 0.2091)
    assert pm.ITRF2020_PMM["PCFC"] == (-0.1122, 0.2836, -0.5984)
    assert pm.ITRF2020_ORB_MM_YR == (0.37, 0.35, 0.74)
    assert len(pm.ITRF2020_PMM) == 13 and set(pm.PLATE_NAMES) == set(pm.ITRF2020_PMM)
    assert "10.1029/2023GL106373" in pm.ITRF2020_PMM_CITATION
    f = PLATE_DIR / "ITRF2020-PMM.dat"
    if not f.exists():
        pytest.skip("the downloaded ITRF2020-PMM.dat is not in review_2026-10-08/p23/plate/")
    assert hashlib.sha256(f.read_bytes()).hexdigest() == pm.ITRF2020_PMM_SHA256
    rows = {}
    for line in f.read_text().splitlines():
        p = line.replace(",", " ").split()
        if len(p) == 4 and p[0].isalpha() and p[0].isupper() and len(p[0]) == 4:
            rows[p[0]] = tuple(float(x) for x in p[1:])
    assert rows == pm.ITRF2020_PMM


def test_plate_velocity_reproduces_the_published_residuals():
    """observed - predicted must equal the residual ITRF published for each
    site; that is only true with the ORB added to the horizontal velocity."""
    for code, plate, lon, lat, ve, vn, re_, rn in PUBLISHED:
        pe, pn = pm.plate_velocity_enu(plate, lat, lon)
        assert ve - pe == pytest.approx(re_, abs=0.05), code
        assert vn - pn == pytest.approx(rn, abs=0.05), code
    # arrays work, and longitudes in -180..180 are the same point
    lat = np.array([r[3] for r in PUBLISHED if r[1] == "AUST"])
    lon = np.array([r[2] for r in PUBLISHED if r[1] == "AUST"]) - 360.0
    pe, pn = pm.plate_velocity_enu("aust", lat, lon)
    assert pe.shape == (4,) and np.allclose(pe, [pm.plate_velocity_enu("AUST", a, o)[0] for a, o in zip(lat, lon)])


def test_the_orb_is_needed_to_match_the_published_model(monkeypatch):
    monkeypatch.setitem(pm.PLATE_MODELS["ITRF2020"], "orb_mm_yr", (0.0, 0.0, 0.0))
    worst = max(max(abs(ve - pm.plate_velocity_enu(p, la, lo)[0] - re_), abs(vn - pm.plate_velocity_enu(p, la, lo)[1] - rn))
                for _, p, lo, la, ve, vn, re_, rn in PUBLISHED)
    assert worst > 0.3


def test_every_published_site_when_the_residual_file_is_here():
    f = PLATE_DIR / "ITRF2020-PMM-residuals.dat"
    if not f.exists():
        pytest.skip("ITRF2020-PMM-residuals.dat is not in review_2026-10-08/p23/plate/")
    names = {v: k for k, v in pm.PLATE_NAMES.items()} | {"N. America": "NOAM", "S. America": "SOAM",
                                                         "Antarctica": "ANTA", "Australia": "AUST",
                                                         "Eurasia": "EURA", "Pacific": "PCFC", "Nubia": "NUBI",
                                                         "Somalia": "SOMA", "Arabia": "ARAB", "India": "INDI",
                                                         "Amurian": "AMUR", "Caribbean": "CARB", "Nazca": "NAZC"}
    d = []
    for line in f.read_text().splitlines():
        plate = line[32:43].strip()
        if plate not in names:
            continue
        lon, lat, ve, vn, _, _, re_, rn = map(float, line[47:].split())
        pe, pn = pm.plate_velocity_enu(names[plate], lat, lon)
        d += [ve - pe - re_, vn - pn - rn]
    d = np.array(d)
    assert d.size == 2 * 518
    assert np.sqrt(np.mean(d ** 2)) < 0.01 and np.abs(d).max() < 0.06


def test_auto_plate_from_where_the_stations_are():
    assert pm.choose_plate(*map(lambda x: [x], APPIN))["plate"] == "AUST"
    for site in (SITE_A, HOUSTON):
        c = pm.choose_plate([site[0]], [site[1]])
        assert c["plate"] == "NOAM" and c["boundary_zone"] is False and "automatic" in c["how"]
    assert pm.choose_plate([POLAND[0]], [POLAND[1]])["plate"] == "EURA"
    # eastern longitudes written as lon - 360 (NGL's habit) are the same place
    assert pm.choose_plate([APPIN[0]], [APPIN[1] - 360.0])["plate"] == "AUST"
    # outside the regions: named, never guessed
    with pytest.raises(ValueError, match="name it"):
        pm.choose_plate([19.72], [-155.05])                            # Hawaii
    with pytest.raises(ValueError, match="no plate in common"):
        pm.choose_plate([APPIN[0], SITE_A[0]], [APPIN[1], SITE_A[1]])
    # named: used as given, case and full name accepted; unknown refused
    assert pm.choose_plate([19.72], [-155.05], plate="pcfc")["plate"] == "PCFC"
    assert pm.choose_plate([0], [0], plate="North American")["plate"] == "NOAM"
    with pytest.raises(ValueError, match="not in ITRF2020-PMM"):
        pm.choose_plate([0], [0], plate="XXXX")


def test_auto_plate_where_two_plates_meet_is_chosen_by_the_stations_velocities():
    lat, lon = CORCORAN
    pe, pn = pm.plate_velocity_enu("NOAM", lat, lon)
    # Great Valley ground: North American plate + ~13 mm/yr north-west (Sierra Nevada-Great Valley block)
    c = pm.choose_plate([lat], [lon], [pe - 10.0], [pn + 8.0])
    assert c["plate"] == "NOAM" and c["boundary_zone"] is True and "Sierra Nevada" in c["note"]
    assert c["candidates"]["NOAM"] < c["candidates"]["PCFC"]
    # Monterey, west of the San Andreas: moves with the Pacific plate
    qe, qn = pm.plate_velocity_enu("PCFC", 36.60, -121.90)
    assert pm.choose_plate([36.60], [-121.90], [qe + 1.0], [qn - 1.0])["plate"] == "PCFC"
    with pytest.raises(ValueError, match="velocities are needed"):
        pm.choose_plate([lat], [lon])
    with pytest.raises(ValueError, match="no station has"):
        pm.choose_plate([lat], [lon], [np.nan], [np.nan])
    with pytest.raises(ValueError, match="different plates"):
        pm.choose_plate([lat, 36.60], [lon, -121.90], [pe, qe], [pn, qn])


def test_plate_los_rate_is_the_horizontal_motion_on_the_line_of_sight():
    pe, pn = pm.plate_velocity_enu("AUST", *APPIN)
    assert pm.plate_los_rate(0.6, -0.1, *APPIN, "AUST") == pytest.approx(0.6 * pe - 0.1 * pn)


# ===================================================================== ground_truth
def _igs20(lat, lon, ve, vn, vu=-5.0, years=3.0, seed=3, frame=True, sigma=1.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=int(years * 365.25), freq="D")
    t = (idx - idx[0]).days.to_numpy() / 365.25
    df = pd.DataFrame({"e_mm": ve * t + rng.normal(0, 1, t.size), "n_mm": vn * t + rng.normal(0, 1, t.size),
                       "u_mm": vu * t + rng.normal(0, 3, t.size), "se_mm": sigma, "sn_mm": sigma, "su_mm": 3.0},
                      index=pd.DatetimeIndex(idx, name="date"))
    df.attrs.update(lat=lat, lon=lon, h=100.0, id="TEST")
    if frame:
        df.attrs.update(frame="IGS20 (NGL daily solutions, as published; plate motion included)", frame_code="IGS20")
    return df


def test_to_plate_frame_removes_the_rigid_plate_motion_and_nothing_else(caplog):
    pe, pn = pm.plate_velocity_enu("AUST", *APPIN, 100.0)
    df = _igs20(*APPIN, pe + 0.5, pn - 0.3)
    out = gt.to_plate_frame(df)
    t = (df.index - df.index[0]).days.to_numpy() / 365.25
    assert np.allclose(out["e_mm"], df["e_mm"] - pe * t) and np.allclose(out["n_mm"], df["n_mm"] - pn * t)
    assert out["e_mm"].iloc[0] == df["e_mm"].iloc[0] and out["n_mm"].iloc[0] == df["n_mm"].iloc[0]  # datum kept
    for col in ("u_mm", "se_mm", "sn_mm", "su_mm"):
        assert out[col].equals(df[col])                                 # Up and sigmas untouched
    e, n = gt.horizontal_rate(out)
    assert e == pytest.approx(0.5, abs=0.1) and n == pytest.approx(-0.3, abs=0.1)
    a = out.attrs
    assert a["frame_code"] == "plate:AUST" and a["frame"].startswith("plate-fixed: Australian plate (AUST)")
    p = a["plate_motion"]
    assert p["removed_mm_yr"] == {"e": pytest.approx(pe, abs=1e-3), "n": pytest.approx(pn, abs=1e-3)}
    assert p["residual_horizontal_mm_yr"]["total"] < 1.0 and p["note"] is None
    assert p["model"] == "ITRF2020-PMM" and "Altamimi" in p["citation"] and p["reference_day"] == "2020-01-01"
    assert p["model_residual_rms_mm_yr"] == {"e": 0.23, "n": 0.22}
    assert df.attrs["frame_code"] == "IGS20" and "plate_motion" not in df.attrs      # the input is not changed
    # a station that does not ride on its plate keeps its motion, with a warning
    caplog.set_level(logging.WARNING, logger="insar_pipeline.ground_truth")
    moving = gt.to_plate_frame(_igs20(*APPIN, pe + 6.0, pn - 4.0))
    assert moving.attrs["plate_motion"]["residual_horizontal_mm_yr"]["total"] == pytest.approx(7.2, abs=0.2)
    assert "does not ride rigidly" in moving.attrs["plate_motion"]["note"] and "does not ride rigidly" in caplog.text


def test_to_plate_frame_one_plate_per_set_and_keeps_missing():
    s = gt.SeriesSet()
    pe, pn = pm.plate_velocity_enu("NOAM", *CORCORAN)
    s["CAHA"] = _igs20(*CORCORAN, pe - 10, pn + 8)
    s["HANF"] = _igs20(36.2152, -119.6591, pe - 9, pn + 7, seed=4)
    s.missing = {"CRCN": "download failed: HTTP 404"}
    out = gt.to_plate_frame(s)
    assert isinstance(out, gt.SeriesSet) and out.missing == s.missing
    assert out.plate["plate"] == "NOAM" and out.plate["boundary_zone"] is True
    assert {df.attrs["frame_code"] for df in out.values()} == {"plate:NOAM"}
    assert gt.to_plate_frame(gt.SeriesSet()) == {}


def test_to_plate_frame_refuses_series_not_in_igs20():
    with pytest.raises(ValueError, match="not IGS20"):
        gt.to_plate_frame(_igs20(*APPIN, 18, 55, frame=False))
    ga = _igs20(*APPIN, 18, 55, frame=False)
    ga.attrs["frame"] = gt.GA_FRAME
    with pytest.raises(ValueError, match="GA CEMP"):
        gt.to_plate_frame(ga)
    already = gt.to_plate_frame(_igs20(*APPIN, 18, 55))
    with pytest.raises(ValueError, match="plate:AUST"):
        gt.to_plate_frame(already)                                       # never removed twice
    nopos = _igs20(*APPIN, 18, 55)
    del nopos.attrs["lat"]
    with pytest.raises(ValueError, match="no station position"):
        gt.to_plate_frame(nopos)


def test_load_gnss_series_plate_frame_end_to_end(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no download expected in this test")
    monkeypatch.setattr(gt.urllib.request, "urlopen", boom)
    cache = tmp_path / "cache"
    ds = G.days(date(2019, 1, 1), date(2022, 12, 31))
    t = G.yrs(ds, ds[0])
    pe, pn = pm.plate_velocity_enu("AUST", -34.2216, 150.6391, 233.9)
    rng = np.random.default_rng(5)
    e = pe * t + rng.normal(0, 1, t.size)
    n = pn * t + rng.normal(0, 1, t.size)
    u = -2.0 * t + rng.normal(0, 3, t.size)
    step = date(2021, 3, 1)
    e = e + np.where(np.array(ds) >= step, 7.0, 0.0)                  # an equipment step, removed as an offset
    G.write_tenv3(cache / "CA07" / "CA07.tenv3", "CA07", ds, e, n, u, lat=-34.2216121985, lon=-209.3608484717,
                  h=233.87602)
    G.write_steps(cache, [("CA07", step, 1)])
    igs = gt.load_gnss_series(["ca07"], None, cache)
    same = gt.load_gnss_series(["CA07"], None, cache, frame="itrf")
    plate = gt.load_gnss_series(["CA07"], None, cache, frame="plate")
    a, p = igs["CA07"], plate["CA07"]
    # default unchanged: IGS20, and 'itrf' is the same thing
    assert a.attrs["frame_code"] == "IGS20" and a.attrs["frame"].startswith("IGS20") and igs.plate is None
    assert same["CA07"].equals(a)
    assert gt.horizontal_rate(a)[1] > 50.0
    # plate-fixed: near zero horizontally, Up identical, the same days, steps and outliers
    he, hn = gt.horizontal_rate(p)
    assert abs(he) < 0.3 and abs(hn) < 0.3
    assert p["u_mm"].equals(a["u_mm"]) and p.index.equals(a.index)
    assert p.attrs["steps_removed"] == a.attrs["steps_removed"] and len(a.attrs["steps_removed"]) == 1
    assert p.attrs["outliers_removed"] == a.attrs["outliers_removed"]
    assert plate.plate["plate"] == "AUST" and p.attrs["frame_code"] == "plate:AUST"
    # IGS14 is still refused; a plate that does not exist fails before anything is read
    with pytest.raises(ValueError, match="only NGL's IGS20"):
        gt.load_gnss_series(["CA07"], None, cache, frame="IGS14")
    with pytest.raises(ValueError, match="not in ITRF2020-PMM"):
        gt.load_gnss_series(["CA07"], None, None, frame="plate", plate="XXXX")
    named = gt.load_gnss_series(["CA07"], None, cache, frame="plate", plate="AUST")
    assert named.plate["how"] == "named in the settings"


# ===================================================================== significance
def test_frame_settings_default_to_plate_and_are_checked():
    assert sg.gnss_frame(SimpleNamespace()) == ("plate", "auto")
    assert sg.gnss_frame(None) == ("plate", "auto")
    assert sg.gnss_frame(SimpleNamespace(reference_gnss_frame="IGS20")) == ("itrf", "auto")
    assert sg.gnss_frame(SimpleNamespace(reference_gnss_frame="ITRF", reference_gnss_plate="noam")) == ("itrf", "NOAM")
    assert sg.gnss_frame(SimpleNamespace(reference_gnss_plate="")) == ("plate", "auto")
    with pytest.raises(ValueError, match="reference_gnss_frame"):
        sg.gnss_frame(SimpleNamespace(reference_gnss_frame="IGS14"))
    with pytest.raises(ValueError, match="reference_gnss_plate"):
        sg.gnss_frame(SimpleNamespace(reference_gnss_plate="Atlantis"))
    base = dict(reference_mode="gnss_absolute", reference_gnss_stations=["A", "B", "C"])
    assert sg.check_reference_settings(SimpleNamespace(**base)) == []
    probs = sg.check_reference_settings(SimpleNamespace(**base, reference_gnss_plate="Atlantis"))
    assert len(probs) == 1 and "reference_gnss_plate" in probs[0]
    assert sg.check_reference_settings(SimpleNamespace(reference_mode="farmland", reference_gnss_frame="x")) == []
    assert {k: v[0] for k, v in sg.REFERENCE_FRAME_SETTINGS.items()} == {"reference_gnss_frame": "plate",
                                                                         "reference_gnss_plate": "auto"}


def test_frame_settings_agree_with_config_wherever_config_declares_them():
    """config.py is another group's file: until it declares the two keys
    they are read with getattr (the defaults above). Once it does, its
    defaults must be these."""
    import dataclasses
    from insar_pipeline import config
    fields = {f.name: f for f in dataclasses.fields(config.PipelineConfig)}
    for key, (default, _) in sg.REFERENCE_FRAME_SETTINGS.items():
        if key in fields:
            assert fields[key].default == default, key


# ===================================================================== deliverables (gnss_absolute)
def _igs20_loader(cfg, truth, **kw):
    """test_p1_absref's fake loader, with its series stated IGS20 and the
    North American plate's motion added to east and north, as NGL's would
    have it."""
    inner = AB._fake_loader(cfg, truth, **kw)

    def load(ids, window, cache_dir):
        out = inner(ids, window, cache_dir)
        for sid, df in out.items():
            pe, pn = pm.plate_velocity_enu("NOAM", df.attrs["lat"], df.attrs["lon"])
            t = (df.index - df.index[0]).days.to_numpy() / 365.25
            df["e_mm"] = df["e_mm"] + pe * t
            df["n_mm"] = df["n_mm"] + pn * t
            df.attrs.update(frame="IGS20 (NGL daily solutions, as published; plate motion included)",
                            frame_code="IGS20")
        return out
    load.calls = inner.calls
    return load


def _abs(tmp_path, monkeypatch, frame=None, **kw):
    cfg, truth = L7a.make_project(tmp_path)
    cfg.reference_mode = "gnss_absolute"
    cfg.reference_gnss_stations = list(AB.STATIONS)
    cfg.reference_gnss_cache_dir = str(tmp_path / "gnss_cache")
    if frame is not None:
        cfg.reference_gnss_frame = frame
    monkeypatch.setattr(dl, "_load_gnss_series", _igs20_loader(cfg, truth, **kw))
    return cfg, truth


def _plate_los(rec):
    """The North American plate's motion on each station's own line of
    sight (the record's los_vector: east, north, up)."""
    out = {}
    for m in rec["stations"]:
        pe, pn = pm.plate_velocity_enu("NOAM", m["lat"], m["lon"])
        le, ln, _ = m["los_vector"]
        assert le == pytest.approx(-np.sin(np.radians(38.5)) * np.sin(np.radians(-100.0)), abs=1e-4)
        out[m["id"]] = le * pe + ln * pn
    return out


@pytest.fixture(autouse=True)
def _clear():
    dl.clear_cache()
    yield
    dl.clear_cache()


def test_gnss_absolute_datum_is_plate_fixed_by_default(tmp_path, monkeypatch):
    cfg, truth = _abs(tmp_path, monkeypatch)
    D = cfg.project_dir / "deliverables"
    _, rel, _, _, _ = dl._empirical_context(cfg)
    rec = json.loads((D / dl.REFERENCE_RECORD).read_text())
    v = truth["vel_mm"]
    ok = np.isfinite(rel)
    # the plate's ~-9 mm/yr on the line of sight is gone: the whole area's -30 mm/yr reads -30
    assert rec["rate"]["offset_mm_yr"] == pytest.approx(-30.0, abs=1e-3)
    assert np.allclose(rel[ok] - v[ok], -30.0, atol=1e-3)
    fr = rec["frame"]
    assert fr["code"] == "plate:NOAM" and fr["name"].startswith("plate-fixed: North American plate (NOAM)")
    assert fr["requested"] == "plate" and fr["plate"]["plate"] == "NOAM" and "automatic" in fr["plate"]["how"]
    assert fr["model"]["sha256"] == pm.ITRF2020_PMM_SHA256 and "Altamimi" in fr["model"]["citation"]
    plos = _plate_los(rec)
    assert all(fr["station_plate_los_rate_mm_yr"][s] == pytest.approx(plos[s], abs=2e-3) for s in AB.STATIONS)
    assert all(abs(x) < 0.05 for x in fr["station_horizontal_los_rate_mm_yr"].values())
    assert all(r["total"] < 0.05 for r in fr["station_residual_horizontal_mm_yr"].values())
    ramp = fr["plate_los_ramp"]
    assert ramp["cells_sampled"] > 1000 and ramp["span_mm_yr"] < 0.05     # one LOS vector, a 3 km grid
    assert "ITRF2020" in fr["note"] and "relative to the plate" in fr["note"]
    assert "relative to the North American plate" in rec["what"]
    assert all(m["plate_removed_mm_yr"]["e"] == pytest.approx(-14.7, abs=0.5) for m in rec["stations"])
    dl.export_measurement_points(cfg, D)
    readme = (D / "points_README.txt").read_text()
    assert "relative to the North American plate" in readme and "reads about zero" in readme
    assert "stable ground does not read zero" not in readme


def test_gnss_absolute_itrf_frame_keeps_the_plate_motion(tmp_path, monkeypatch):
    cfg, truth = _abs(tmp_path, monkeypatch, frame="itrf")
    D = cfg.project_dir / "deliverables"
    dl._empirical_context(cfg)
    rec = json.loads((D / dl.REFERENCE_RECORD).read_text())
    plos = np.mean(list(_plate_los(rec).values()))
    assert plos == pytest.approx(-9.0, abs=0.5)                        # Indiana, ascending: about -9 mm/yr
    assert rec["rate"]["offset_mm_yr"] == pytest.approx(-30.0 + plos, abs=0.01)
    fr = rec["frame"]
    assert fr["code"] == "IGS20" and fr["name"] == "IGS20, as NGL publishes the GNSS" and fr["requested"] == "itrf"
    assert fr["station_horizontal_los_rate_median_mm_yr"] == pytest.approx(plos, abs=0.05)
    dl.export_measurement_points(cfg, D)
    readme = (D / "points_README.txt").read_text()
    assert "IGS20" in readme and "stable ground does not read zero" in readme


def test_gnss_absolute_series_that_state_no_frame_are_not_relabelled(tmp_path, monkeypatch):
    """test_p1_absref's loader states no frame: nothing is removed and the
    record says so in words (never a plate label on IGS20 numbers)."""
    cfg, truth = L7a.make_project(tmp_path)
    cfg.reference_mode = "gnss_absolute"
    cfg.reference_gnss_stations = list(AB.STATIONS)
    cfg.reference_gnss_cache_dir = str(tmp_path / "gnss_cache")
    monkeypatch.setattr(dl, "_load_gnss_series", AB._fake_loader(cfg, truth))
    dl._empirical_context(cfg)
    fr = json.loads((cfg.project_dir / "deliverables" / dl.REFERENCE_RECORD).read_text())["frame"]
    assert fr["code"] == "IGS20" and fr["name"].startswith("IGS20 assumed") and "not removed" in fr["reason"]


def test_gnss_absolute_refuses_a_datum_of_mixed_frames(tmp_path, monkeypatch):
    cfg, truth = _abs(tmp_path, monkeypatch)
    loader = dl._load_gnss_series

    def mixed(ids, window, cache_dir):
        out = loader(ids, window, cache_dir)
        out["BBBB"].attrs.pop("frame_code")
        out["BBBB"].attrs.pop("frame")
        return out
    monkeypatch.setattr(dl, "_load_gnss_series", mixed)
    with pytest.raises(ValueError, match="one datum needs one frame"):
        dl._empirical_context(cfg)
    rec = json.loads((cfg.project_dir / "deliverables" / dl.REFERENCE_RECORD).read_text())
    assert rec["available"] is False and "one frame" in rec["reason"]


def test_gnss_station_series_frame_arguments_override_the_config(tmp_path, monkeypatch):
    cfg, _ = _abs(tmp_path, monkeypatch, frame="itrf")
    S = dl.gnss_station_series(cfg, frame="plate", plate="NOAM")
    assert S["frame"]["applied"] is True and S["frame"]["plate"]["how"] == "named in the settings"
    S2 = dl.gnss_station_series(cfg)
    assert S2["frame"]["applied"] is False and S2["frame"]["frame"] == "IGS20"
    d = (S2["gnss_los_mm"] - S["gnss_los_mm"])[:, -1] / S["t"][-1]
    assert np.allclose(d, [m["plate_los_rate_mm_yr"] for m in S["stations"]], atol=0.01)
    with pytest.raises(ValueError, match="not one of"):
        dl.gnss_station_series(cfg, frame="IGS14")


# ===================================================================== real data (read-only)
def _real_cache(tmp_path, sites):
    if not all((P1_CACHE / s / f"{s}.tenv3").exists() for s in sites) or not (P1_CACHE / "ngl_steps.txt").exists():
        pytest.skip("the phase 1 GNSS cache (review_2026-10-08/p1/cache) is not here")
    c = tmp_path / "cache"
    for s in sites:
        shutil.copytree(P1_CACHE / s, c / s)
    shutil.copy(P1_CACHE / "ngl_steps.txt", c / "ngl_steps.txt")
    return c


@pytest.mark.parametrize("site,plate,igs20_min", [("CA07", "AUST", 50.0), ("LAMC", "NOAM", 14.0)])
def test_real_stable_stations_read_near_zero_horizontally_after_removal(tmp_path, monkeypatch, site, plate,
                                                                        igs20_min):
    """CA07 (Appin's reference station, Australia, 2016-2026) and LAMC
    (Indiana, stable North America, 2008-2018), NGL's IGS20 series as cached
    in phase 1: 57 and 15 mm/yr horizontally in IGS20, under 1 mm/yr
    relative to their plates (8 Oct 2026: CA07 -0.30 E +0.78 N, LAMC +0.23 E
    +0.30 N)."""
    def boom(*a, **k):
        raise AssertionError("no download expected in this test")
    monkeypatch.setattr(gt.urllib.request, "urlopen", boom)
    c = _real_cache(tmp_path, [site])
    a = gt.load_gnss_series([site], None, c, refresh=False, earthquake_steps=True)[site]
    p = gt.load_gnss_series([site], None, c, frame="plate", refresh=False, earthquake_steps=True)
    assert float(np.hypot(*gt.horizontal_rate(a))) > igs20_min
    r = p[site].attrs["plate_motion"]["residual_horizontal_mm_yr"]
    assert p.plate["plate"] == plate and r["total"] < 1.0, r
    assert p[site]["u_mm"].equals(a["u_mm"])


def test_real_corcoran_station_keeps_its_great_valley_motion(tmp_path, monkeypatch):
    """CAHA (Corcoran) is in the plate-boundary zone: 'auto' chooses North
    America by its velocity, and its ~15 mm/yr north-west motion relative to
    stable North America (the Sierra Nevada-Great Valley block) stays, with
    the warning."""
    def boom(*a, **k):
        raise AssertionError("no download expected in this test")
    monkeypatch.setattr(gt.urllib.request, "urlopen", boom)
    c = _real_cache(tmp_path, ["CAHA"])
    p = gt.load_gnss_series(["CAHA"], None, c, frame="plate", refresh=False, earthquake_steps=True)
    r = p["CAHA"].attrs["plate_motion"]["residual_horizontal_mm_yr"]
    assert p.plate["plate"] == "NOAM" and p.plate["boundary_zone"] is True
    assert 8.0 < r["total"] < 20.0 and r["e"] < 0 < r["n"]
    assert "does not ride rigidly" in p["CAHA"].attrs["plate_motion"]["note"]


# ===================================================================== follow-up checks (8 Oct 2026, second pass)
@pytest.mark.parametrize("name,lat,lon,plate", [
    ("Perth", -31.95, 115.86, "AUST"), ("Hobart", -42.88, 147.33, "AUST"), ("Tahmoor", -34.25, 150.59, "AUST"),
    ("Darwin", -12.46, 130.84, "AUST"), ("Cape York tip", -10.69, 142.53, "AUST"), ("Gove", -12.27, 136.82, "AUST"),
    ("Kalumburu", -14.30, 126.64, "AUST"), ("Lyon", 45.76, 4.84, "EURA"), ("Dublin", 53.35, -6.26, "EURA"),
    ("Dingle", 52.14, -10.27, "EURA"), ("Berlin", 52.52, 13.40, "EURA"), ("Bern", 46.95, 7.45, "EURA"),
    ("Rydultowy", 50.05, 18.42, "EURA"), ("Site D, Indiana", 38.95, -87.30, "NOAM")])
def test_auto_plate_regions_hold_what_they_claim(name, lat, lon, plate):
    assert pm.choose_plate([lat], [lon])["plate"] == plate, name


@pytest.mark.parametrize("name,lat,lon", [
    ("Sumba (Banda arc)", -9.70, 120.00), ("Kupang, West Timor", -10.17, 123.60), ("Rote", -10.75, 123.10),
    ("Dili, Timor-Leste", -8.56, 125.57), ("Port Moresby", -9.44, 147.18), ("Merauke", -8.50, 140.40),
    ("Milan (Po plain, Adria)", 45.46, 9.19), ("Venice", 45.44, 12.33), ("Ljubljana", 46.05, 14.51)])
def test_auto_plate_is_not_guessed_where_plates_meet_outside_the_regions(name, lat, lon):
    """The first boxes reached 9 S (so Sumba, Rote and Timor, on the Banda
    arc, read Australian) and 45 N across Europe (so the Po plain, on Adria,
    read Eurasian). There the plate must be named."""
    with pytest.raises(ValueError, match="name it"):
        pm.choose_plate([lat], [lon])


def test_short_record_warning_says_the_annual_cycle_may_set_the_rate(caplog):
    pe, pn = pm.plate_velocity_enu("NOAM", *SITE_A, 100.0)
    caplog.set_level(logging.WARNING, logger="insar_pipeline.ground_truth")
    short = gt.to_plate_frame(_igs20(*SITE_A, pe - 4.0, pn + 5.5, years=0.9))
    p = short.attrs["plate_motion"]
    assert p["residual_horizontal_mm_yr"]["span_yr"] == pytest.approx(0.9, abs=0.01)
    assert "only 0.9 yr" in p["note"] and "not yet evidence" in p["note"] and "does not ride rigidly" not in p["note"]
    long_ = gt.to_plate_frame(_igs20(*SITE_A, pe - 4.0, pn + 5.5, years=gt.PLATE_RESIDUAL_MIN_SPAN_YR + 0.1))
    assert "does not ride rigidly" in long_.attrs["plate_motion"]["note"]
    assert long_.attrs["plate_motion"]["residual_horizontal_mm_yr"]["span_yr"] >= gt.PLATE_RESIDUAL_MIN_SPAN_YR


def test_to_plate_frame_of_an_empty_series_does_not_fail():
    df = _igs20(*APPIN, 18, 55).iloc[:0]
    out = gt.to_plate_frame(df)
    assert len(out) == 0 and out.attrs["frame_code"] == "plate:AUST"
    assert out.attrs["plate_motion"]["residual_horizontal_mm_yr"] is None
    assert out.attrs["plate_motion"]["reference_day"] is None


def test_gnss_in_frame_with_no_series_says_so():
    series, rec = dl._gnss_in_frame(SimpleNamespace(), gt.SeriesSet())
    assert rec["applied"] is False and rec["frame"] is None and "no GNSS series" in rec["reason"]
    assert rec["requested"] == "plate" and series == {}


def test_real_short_record_station_is_not_called_off_its_plate(tmp_path, monkeypatch):
    """FTBR (Indiana) has 0.8 yr of record: its 7 mm/yr straight-line rate
    relative to the plate is the annual cycle's, and the note says so."""
    def boom(*a, **k):
        raise AssertionError("no download expected in this test")
    monkeypatch.setattr(gt.urllib.request, "urlopen", boom)
    c = _real_cache(tmp_path, ["FTBR"])
    p = gt.load_gnss_series(["FTBR"], None, c, frame="plate", refresh=False, earthquake_steps=True)
    pmo = p["FTBR"].attrs["plate_motion"]
    assert pmo["residual_horizontal_mm_yr"]["span_yr"] < 1.0
    assert "not yet evidence" in pmo["note"] and "does not ride rigidly" not in pmo["note"]


# NGL's IGS20 MIDAS tables (gps_timeseries/IGS20/midas/). ngl/ beside it holds the /velocities/ tables, which
# NGL says are IGS14 (superseded): ITRF2020-PMM is not checked against those.
NGL_DIR = PLATE_DIR / "ngl_igs20"
_MIDAS_COLS = ["id", "ver", "t0", "t1", "span", "nep", "ngood", "npair", "ve", "vn", "vu", "se", "sn", "su", "e0",
               "n0", "u0", "oe", "on", "ou", "sde", "sdn", "sdu", "nsteps", "lat", "lon", "h"]


def _midas(name):
    f = NGL_DIR / name
    if not f.exists():
        pytest.skip(f"{name} (NGL's IGS20 MIDAS velocities) is not in review_2026-10-08/p23/plate/ngl_igs20/")
    d = pd.read_csv(f, sep=r"\s+", header=None, names=_MIDAS_COLS, usecols=range(27))
    d["lon"] = (d["lon"] + 180.0) % 360.0 - 180.0
    return d.set_index("id")


@pytest.mark.parametrize("plate,fixed,lat,lon,n_min", [("NOAM", "midas.NA.txt", (30, 45), (-100, -75), 2000),
                                                       ("AUST", "midas.AU.txt", (-44, -10), (113, 154), 500)])
def test_real_stable_networks_read_near_zero_relative_to_their_plate(plate, fixed, lat, lon, n_min):
    """Every NGL station with 5+ years and a rate sigma under 0.5 mm/yr in
    stable North America (30-45 N, 100-75 W) and in Australia, IGS20 MIDAS
    rates: relative to the plate the median is within 0.2 mm/yr of zero per
    component and 85% are under 1 mm/yr (8 Oct 2026: 2366 stations,
    -0.07/-0.01, 91%; 648 stations, -0.02/-0.15, 88%). NGL's own plate-fixed
    velocities agree to 0.2 mm/yr in Australia; NGL's North America frame
    (its own pole) sits 0.8 mm/yr north of this one."""
    igs = _midas("midas.IGS.txt")
    assert igs.t1.max() > 2025.0                  # NGL's live IGS20 table, not the IGS14 one frozen at 2024.65
    s = igs[igs.lat.between(*lat) & igs.lon.between(*lon) & (igs.span >= 5) & (igs.se < 5e-4) & (igs.sn < 5e-4)]
    assert len(s) >= n_min
    pe, pn = pm.plate_velocity_enu(plate, s.lat.to_numpy(), s.lon.to_numpy(), s.h.to_numpy())
    re_, rn = s.ve.to_numpy() * 1000 - pe, s.vn.to_numpy() * 1000 - pn
    assert np.median(np.hypot(s.ve, s.vn)) * 1000 > 14.0                    # IGS20: the plate's motion is there
    assert abs(np.median(re_)) < 0.2 and abs(np.median(rn)) < 0.2
    assert np.mean(np.hypot(re_, rn) < 1.0) > 0.85
    fx = _midas(fixed).reindex(s.index)
    ok = fx.ve.notna().to_numpy()
    de = re_[ok] - fx.ve.to_numpy()[ok] * 1000
    dn = rn[ok] - fx.vn.to_numpy()[ok] * 1000
    if plate == "AUST":
        assert abs(np.median(de)) < 0.2 and abs(np.median(dn)) < 0.2
    else:
        assert abs(np.median(de)) < 0.2 and -1.0 < np.median(dn) < -0.5


def test_real_ca07_residual_is_the_stations_own():
    """CA07 reads under 1 mm/yr relative to the Australian plate (IGS20
    MIDAS: -0.15 east, +0.57 north), and its north residual is in NGL's own
    Australia-fixed MIDAS rate too (+0.63), so it is the station's, not this
    model's; Port Kembla (PTKL, 25 km away) reads -0.06 / 0.00. The two
    frames differ by the same amount at both stations (about -0.17 east,
    -0.06 north: the poles', constant across the basin), so what is left
    at CA07 is the station's."""
    igs, au = _midas("midas.IGS.txt"), _midas("midas.AU.txt")
    diff = {}
    for sid, total_max in [("CA07", 1.0), ("PTKL", 0.3)]:
        r = igs.loc[sid]
        pe, pn = pm.plate_velocity_enu("AUST", r.lat, r.lon, r.h)
        re_, rn = r.ve * 1000 - pe, r.vn * 1000 - pn
        assert np.hypot(re_, rn) < total_max, sid
        diff[sid] = np.array([re_ - au.loc[sid].ve * 1000, rn - au.loc[sid].vn * 1000])
        assert np.all(np.abs(diff[sid]) < 0.3), sid
    assert np.all(np.abs(diff["CA07"] - diff["PTKL"]) < 0.05)


# ===================================================================== reviewer's checks (8 Oct 2026, third pass)
def _frame_S(ramp=True):
    return {"stations": [{"id": "AAAA", "usable": True, "horizontal_los_rate_mm_yr": 0.1}],
            "frame": {"applied": True, "requested": "plate", "frame": "plate:NOAM",
                      "plate": {"plate": "NOAM", "name": "North American", "boundary_zone": False},
                      "plate_los_ramp": {"cells_sampled": 100, "every_nth_cell": 1,
                                         "stations_mean_plate_los_rate_mm_yr": 8.9, "min_mm_yr": -0.2,
                                         "max_mm_yr": 0.2, "span_mm_yr": 0.4} if ramp else None}}


@pytest.mark.parametrize("deramp,plane,words", [
    ("linear", False, "linear deramp, which takes a ramp like this out"),
    ("no", True, "the datum's plane (reference_gnss_plane) takes the linear part of it out"),
    ("no", False, "no deramp and no datum plane, so it is in the product")])
def test_frame_record_says_what_takes_the_plate_ramp_out(deramp, plane, words):
    """The datum's per-date plane (reference_gnss_plane) is fitted to the
    stations' GNSS minus InSAR, so it takes out the near-linear plate ramp
    between them; the record said 'it is in the product' whenever there was
    no deramp, even with the plane on."""
    rec = dl._frame_record(SimpleNamespace(mintpy_deramp=deramp, reference_gnss_plane=plane), _frame_S())
    assert words in rec["note"] and rec["datum_plane"] is plane
    assert "it is in the product" not in rec["note"] or (deramp == "no" and not plane)


def test_plate_los_ramp_size_over_a_sentinel1_scene():
    """The size significance.reference_mode's docstring states: with
    Sentinel-1 the plate's line-of-sight rate changes by 0.3-0.7 mm/yr across
    a 30 km box (incidence 38 deg +- 1.9 deg across range), the most at Appin
    ascending (-17 mm/yr at the centre); the plate velocity's own change
    over 30 km is about 0.2 mm/yr in Australia and 0.05 in Indiana."""
    def los(inc, az):
        i, a = np.radians(inc), np.radians(az)
        return np.sin(i) * np.sin(a), np.sin(i) * np.cos(a)
    x = np.linspace(-15.0, 15.0, 31)
    X, Y = np.meshgrid(x, x)
    spans = {}
    for site, (lat0, lon0), plate in (("Appin", APPIN, "AUST"), ("Site A", SITE_A, "NOAM")):
        lat, lon = lat0 + Y / 111.0, lon0 + X / (111.0 * np.cos(np.radians(lat0)))
        for name, az, sign in (("asc", 260.0, 1.0), ("desc", 100.0, -1.0)):
            le, ln = los(38.0 + sign * X * 15.5 / 250.0, az)
            p = pm.plate_los_rate(le, ln, lat, lon, plate)
            spans[site, name] = (float(p[15, 15]), float(np.ptp(p)))
            only_plate = pm.plate_los_rate(*los(38.0, az), lat, lon, plate)
            assert np.ptp(only_plate) == pytest.approx(0.2 if plate == "AUST" else 0.05, abs=0.04)
    assert spans["Appin", "asc"][0] == pytest.approx(-17.0, abs=0.5)
    assert spans["Site A", "asc"][0] == pytest.approx(8.9, abs=0.5)
    assert all(0.25 < s < 0.75 for _, s in spans.values()) and max(s for _, s in spans.values()) > 0.6
