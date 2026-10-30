"""
Group E_qa (7 Oct 2026 fixes): the quality assessment's send rule,
geolocation verdict, coherence-bias check, workings total, resolution check
and date accounting. Each test fails on the code in
backups/pre_fixes_20261007/ and passes now. No network.

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_E_qa.py -q -p no:cacheprovider
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from insar_pipeline import quality_assessment as qa                 # noqa: E402
from insar_pipeline import stack_qc                                  # noqa: E402
from insar_pipeline.config import PipelineConfig                    # noqa: E402

import test_fix_L8 as L8                                             # noqa: E402  (the QA project builder)

CRASH = {"available": False, "reason": "could not run: MemoryError()", "flag": None}
CLEAN = {"checks": {"coverage": {"pass": True}}, "geolocation": {"flag": False}}


@pytest.fixture(autouse=True)
def no_osm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the QA tried to fetch OSM water in a test")
    monkeypatch.setattr(stack_qc, "fetch_osm_water", boom, raising=False)


# ------------------------------------------------------------------ #19 a diagnostic that raised is not a pass
@pytest.mark.parametrize("key", ["ramp_sensitivity", "matching_sensitivity", "plausibility", "dem_error"])
def test_a_crashed_diagnostic_that_can_hold_holds(key):
    assert qa.send_rule(CLEAN)["value"] == "send"
    sr = qa.send_rule(dict(CLEAN, **{key: dict(CRASH)}))
    assert sr["value"] == "hold for specialist"
    assert any("could not run" in h and "MemoryError" in h for h in sr["hold_reasons"]), sr


def test_a_crashed_note_only_diagnostic_is_a_note():
    for key in ("resolution_check", "reference_distance"):
        sr = qa.send_rule(dict(CLEAN, **{key: dict(CRASH)}))
        assert sr["value"] == "send with notes" and any("could not run" in n for n in sr["notes"]), key
    for sub in ("disturbed", "second_gate"):
        cov = {"pass": True}
        if sub == "disturbed":
            cov["disturbed"] = dict(CRASH)
        else:
            cov["second_gate"] = {"effect_on_figures": dict(CRASH)}
        sr = qa.send_rule(dict(CLEAN, checks={"coverage": cov}))
        assert sr["value"] == "send with notes" and any("could not run" in n for n in sr["notes"]), sub
    # a diagnostic that does not apply (no layer) stays silent, as before
    na = {"available": False, "reason": "no mapped-mine layer (deramp_exclude_geojson)", "flag": None}
    assert qa.send_rule(dict(CLEAN, reference_distance=na, ramp_sensitivity=na))["value"] == "send"


def test_evaluate_holds_when_the_ramp_diagnostic_raises(tmp_path, monkeypatch):
    def boom(ctx):
        raise MemoryError("synthetic")
    monkeypatch.setattr(qa, "_ramp_sensitivity", boom)
    r = qa.evaluate(L8.build(tmp_path))
    assert r["ramp_sensitivity"]["reason"].startswith("could not run")
    assert any("ramp sensitivity check could not run" in h for h in r["send_rule"]["hold_reasons"])


# ------------------------------------------------------------------ #20 geolocation: worse offsets are never milder
def _geo(off, west, east, dis):
    return {"available": True, "n_water_bodies": 30, "step_m": 2.5, "offset_m": off,
            "halves": {"west": {"offset_m": west}, "east": {"offset_m": east}},
            "west_east_disagreement_m": dis}


def test_a_large_joint_offset_is_flagged_even_when_the_halves_disagree():
    g = qa.judge_geolocation(_geo(30.0, 45.0, 12.0, 35.0))
    assert g["flag"] is True and "45 m off" in g["text"] and "within" not in g["text"]
    assert qa.judge_geolocation(g) == g                                   # idempotent
    sr = qa.send_rule({"checks": {}, "geolocation": g})
    assert sr["value"] == "hold for specialist" and any("45.0 m off" in h for h in sr["hold_reasons"])
    # E22 (trust review 7 Oct 2026): a 12 m half is within one 15 m cell -
    # not flagged (it held the report until then)
    g = qa.judge_geolocation(_geo(5.0, 12.0, 3.0, 9.0))
    assert g["flag"] is False
    # the hold text names the worst offset, not the 5 m joint one: a half over
    # a cell that the split test says differs beyond the halves' own scatter
    g = qa.judge_geolocation(dict(_geo(5.0, 21.0, 3.0, 18.0), split_test={"p_value": 0.01}))
    assert g["flag"] is True
    assert any("21.0 m off" in h for h in qa.send_rule({"checks": {}, "geolocation": g})["hold_reasons"])


def test_an_inconclusive_check_with_a_half_over_10_m_holds():
    import test_fix_final as F
    g = qa.judge_geolocation(F.SITE_B_GEO)
    assert g["flag"] is None and g["status"].startswith("inconclusive") and "within about" not in g["text"]
    assert g["hold"] is True and g["worst_offset_m"] == 21.54
    assert qa.judge_geolocation(g) == g
    sr = qa.send_rule({"checks": {}, "geolocation": g})
    assert sr["value"] == "hold for specialist" and any("21.54 m" in h for h in sr["hold_reasons"])
    # halves disagree but neither is over one cell: E22 (7 Oct 2026) - not
    # inconclusive any more, the whole-area figure stands
    g = qa.judge_geolocation(_geo(3.0, 9.0, 1.0, 12.0))
    assert g["flag"] is False and not g.get("hold")
    assert qa.send_rule({"checks": {}, "geolocation": g})["value"] != "hold for specialist"


# ------------------------------------------------------------------ #23 the coherence-bias check above a 0.6 threshold
def test_ds_bias_is_measured_at_a_0_7_threshold(tmp_path):
    rng = np.random.default_rng(7)
    tc = rng.uniform(0.7, 1.0, (L8.NY, L8.NX))
    r = qa.evaluate(L8.build(tmp_path, tc=tc, thr=0.7))
    ds = r["checks"]["distributed_scatterer_bias"]
    assert ds["pass"] is not None and ds["low_class"] == "0.7-0.8" and ds["high_class"] == "0.9-1.0"
    assert not any("distributed scatterer bias check could not be run" in h for h in r["send_rule"]["hold_reasons"])


def test_ds_bias_with_one_class_does_not_apply_and_does_not_hold(tmp_path):
    r = qa.evaluate(L8.build(tmp_path, tc=np.full((L8.NY, L8.NX), 0.95), thr=0.7))
    ds = r["checks"]["distributed_scatterer_bias"]
    assert ds["pass"] is None and "0.9-1.0" in ds["not_applicable"]       # untested, never a pass
    sr = r["send_rule"]
    assert not any("distributed scatterer bias" in h for h in sr["hold_reasons"])
    assert any("distributed scatterer bias check does not apply" in n for n in sr["notes"])
    # with no class at all it is still an untested check that holds
    assert qa.send_rule({"checks": {"distributed_scatterer_bias": {"pass": None}}})["value"] == "hold for specialist"


# ------------------------------------------------------------------ #24 an unmeasured noise floor is said
def test_unmeasured_noise_floor_on_a_phase_linked_route_is_noted():
    cov = {"pass": True, "threshold": 0.4, "threshold_above_noise": None,
           "noise_level": {"measured": False, "floor": None, "population": "no unwrapping-consistency layer"},
           "coherence_source": "MiaplPy phase linking, average (mini-stack) temporal coherence"}
    sr = qa.send_rule({"checks": {"coverage": cov}, "geolocation": {"flag": False}})
    assert sr["value"] == "send with notes"
    assert any("not checked against a measured noise floor" in n for n in sr["notes"]), sr
    # the classic route grades on MintPy's own coherence: no floor to measure, nothing to say
    classic = dict(cov, coherence_source="MintPy network-inversion temporal coherence",
                   noise_level={"measured": False, "floor": None, "population": "the working coherence is MintPy's own"})
    assert qa.send_rule({"checks": {"coverage": classic}, "geolocation": {"flag": False}})["value"] == "send"


# ------------------------------------------------------------------ #25 the workings total is the measured change
def test_workings_total_is_the_measured_change(tmp_path, monkeypatch):
    # three years at 24 days; the mapped panel sinks 30 mm in its first year, then stops
    from datetime import date, timedelta
    dates = [(date(2018, 6, 11) + timedelta(days=24 * i)).strftime("%Y%m%d") for i in range(46)]
    monkeypatch.setattr(L8, "DATES", dates)
    cfg = L8.build(tmp_path, vel_mm=np.zeros((L8.NY, L8.NX)))
    lon, lat = L8._lonlat()
    mine = (lon > -87.70) & (lon < -87.66) & (lat > 38.40) & (lat < 38.43)
    t = np.arange(len(dates)) * 24 / 365.25
    path = cfg.mintpy_dir / "geo" / "geo_timeseries_ERA5_ramp_demErr.h5"
    with h5py.File(path, "r+") as f:
        ts = f["timeseries"][:]
        ts[:, mine] += (-0.030 * np.minimum(t, 1.0))[:, None].astype(np.float32)
        f["timeseries"][:] = ts
    w = qa.evaluate(cfg)["findings"]["workings_relative_to_far_field"]
    assert w["total_method"] != "rate_x_span" and "total_at_average_rate_mm" in w
    assert abs(w["total_mm"] - (-30.0)) < 5.0                             # what the ground did
    assert abs(w["total_at_average_rate_mm"] - w["total_mm"]) > 5.0       # rate x span says otherwise


# ------------------------------------------------------------------ #26 the resolution check's worst case flags
def _res_ctx(tmp_path, corr_px, n=1024, seed=4):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1.0, (n, n))
    ky = np.fft.fftfreq(n)[:, None]; kx = np.fft.rfftfreq(n)[None, :]
    vel = np.fft.irfft2(np.fft.rfft2(noise) * np.exp(-2 * (np.pi * corr_px) ** 2 * (kx ** 2 + ky ** 2)), s=(n, n))
    vel = (vel / vel.std()).astype(np.float32)
    attrs = {"X_FIRST": "400000", "Y_FIRST": "4200000", "X_STEP": "15", "Y_STEP": "-15",
             "LENGTH": str(n), "WIDTH": str(n)}
    cfg = PipelineConfig(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.5, 38.5],
                         start_date="2018-06-01", end_date="2019-06-30", backend="miaplpy")
    good = np.ones((n, n), bool)
    return {"cfg": cfg, "vel": vel, "tc": np.full((n, n), 0.8, np.float32), "good": good, "attrs": attrs}


def test_errors_shared_beyond_the_measured_window_are_flagged(tmp_path, monkeypatch):
    from insar_pipeline import resolution
    monkeypatch.setattr(resolution, "effective_resolution", lambda cfg, attrs=None: {"support_m": [90, 240],
                                                                                     "text": "90 x 240 m"})
    r = qa._resolution_check(_res_ctx(tmp_path, corr_px=100))           # errors shared over ~1.5 km
    assert r["available"] and r["half_correlation_is_lower_bound"]["east_west"] is True
    assert r["half_correlation_distance_m"]["east_west"] >= 500 and r["flag"] is True
    # a lower bound short of twice the support decides nothing
    monkeypatch.setattr(resolution, "effective_resolution", lambda cfg, attrs=None: {"support_m": [400, 400],
                                                                                     "text": "400 m"})
    assert qa._resolution_check(_res_ctx(tmp_path, corr_px=100))["flag"] is None
    # short correlation: measured, within twice the support, not flagged
    r = qa._resolution_check(_res_ctx(tmp_path, corr_px=1))
    assert r["half_correlation_is_lower_bound"]["east_west"] is False and r["flag"] is False


# ------------------------------------------------------------------ #27 every catalogue date is accounted for
def _dates_cfg(tmp_path, **over):
    kw = dict(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.52, 38.47], start_date="2019-06-01",
              end_date="2020-06-30", backend="miaplpy")
    kw.update(over)
    cfg = PipelineConfig(**kw)
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    cfg.slc_dir.mkdir(parents=True, exist_ok=True)
    return cfg


PROCESSED = ["20191208", "20191220", "20200125", "20200206"]


def test_catalogue_dates_never_assembled_are_named(tmp_path):
    cfg = _dates_cfg(tmp_path)
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps(
        {"catalogue_dates": PROCESSED + ["20200101", "20200113", "20200901"],
         "accepted_not_in_plan": ["20200101"], "accepted_missing": [], "failed": {}}))
    r = qa.date_accounting(cfg, PROCESSED)
    assert "processing plan" in r["dropped"]["20200101"] and "accepted" in r["dropped"]["20200101"]
    assert r["unexplained"] == ["20200113"] and "unexplained" in r["dropped"]["20200113"]
    assert "outside the configured date range" in r["dropped"]["20200901"]
    assert r["processed"] + len(r["dropped"]) == r["available"] == 7
    assert r["hold"] is True
    assert any("20200113" in h for h in qa.send_rule({"checks": {}, "dates": r})["hold_reasons"])


def test_a_catalogue_date_the_config_accepts_is_not_unexplained(tmp_path):
    cfg = _dates_cfg(tmp_path, allow_missing_dates=("2020-01-13",))
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps(
        {"catalogue_dates": PROCESSED + ["20200113"], "failed": {}}))
    r = qa.date_accounting(cfg, PROCESSED)
    assert r["unexplained"] == [] and "accepted as missing" in r["dropped"]["20200113"]
    assert r["processed"] + len(r["dropped"]) == r["available"]


def test_an_assembled_date_stays_unexplained_even_when_accepted(tmp_path):
    """the 20161206 case: on disk but never in the stack is a pipeline fault,
    whatever the config accepts as missing from the archive"""
    cfg = _dates_cfg(tmp_path, allow_missing_dates=("20200113",))
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps(
        {"catalogue_dates": PROCESSED + ["20200113"], "failed": {}}))
    (cfg.slc_dir / "S1A_IW_SLC__1SDV_20200113T235959_20200114T000026_030000_037000_ABCD.SAFE").mkdir()
    r = qa.date_accounting(cfg, PROCESSED)
    assert r["unexplained"] == ["20200113"] and r["dropped"]["20200113"].startswith(qa._NEVER_ENTERED)


def test_a_not_in_plan_date_is_not_called_short(tmp_path):
    """Review: data_acquisition now records a date the plan cannot take in
    assembly_status 'dropped' with the code 'not_in_plan'; an unknown code
    read as 'the archive search returned it incomplete'."""
    cfg = _dates_cfg(tmp_path, allow_missing_dates=("20200101",))
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps(
        {"catalogue_dates": PROCESSED + ["20200101"], "dropped": {"20200101": "not_in_plan"},
         "accepted_not_in_plan": ["20200101"], "failed": {}}))
    r = qa.date_accounting(cfg, PROCESSED)
    why = r["dropped"]["20200101"]
    assert "processing plan" in why and "incomplete" not in why and "accepted as missing" in why
    assert r["unexplained"] == [] and r["processed"] + len(r["dropped"]) == r["available"]
