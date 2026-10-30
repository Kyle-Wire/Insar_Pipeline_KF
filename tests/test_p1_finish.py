"""Phase 1 finishing pass (review_2026-10-08/PLAN.md section 3, 8 Oct 2026):
the five phase 1 pieces wired together.

  * the experiment configs name purpose: experiment, their site_profile and
    the fixed protocol whose `runs:` list names them, and say nothing more
    about "until V3";
  * the B1 reference keys are config settings (load_config used to reject
    them as unknown) and the config check refuses unusable values;
  * `insar validate` reaches the harness from the dashboard's command line,
    and the dashboard names the validation stage;
  * a products run of an experiment calls the REAL harness, which writes its
    records under <project>/validation/protocol_<id>_<run>;
  * decompose.py's own output (not an imitation of it) is what the harness's
    --up reads, on its own grid and epochs;
  * the harness removes the steps the protocol says (steps.txt: equipment
    and earthquake), maps protocol roles exactly as ground_truth does, uses
    the B1 code's joint datum fit as a plane sensitivity, and warns when the
    product's own GNSS datum was fitted to a station it scores.

Each behaviour test fails on the tree before this pass. Three are guards
on what already worked and must keep working: the fixed protocols are
untouched, decompose.py's real output reaches the harness (only an
imitation of it was tested), and the harness's offset datum equals the B1
fit. Synthetic inputs only: no network, nothing under ~/insar_projects.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from insar_pipeline import config, ground_truth as gt, notify, pipeline, procs, progress, runs   # noqa: E402
from insar_pipeline import significance as sg, validation as V                                 # noqa: E402
from insar_pipeline.config import ConfigError, load_config                                     # noqa: E402
import test_p1_decomp as D                                                                    # noqa: E402  (synthetic runs)
import test_p1_harness as H                                                                   # noqa: E402  (synthetic product, FakeGT)

PKG = Path(__file__).resolve().parents[1]
EXP = PKG / "configs" / "experiments"
REVIEW = PKG / "review_2026-10-08"
@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("INSAR_NOTIFY", "off")
    monkeypatch.setattr(notify, "LOGFILE", tmp_path / "notifications.log")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    yield
    procs.clear_stopping()


# ====================================================================== configs

def test_the_fixed_protocols_are_untouched():
    text = (EXP / "README.md").read_text()
    for p in ("protocol_A.yaml", "protocol_B1.yaml", "protocol_B2.yaml"):
        assert V.fixed_hash_status(REVIEW / p)["status"] == "fixed", p
        assert V._sha256(REVIEW / p) in text, p


# ====================================================================== B1 keys are settings

def _yaml(tmp_path, extra=""):
    f = tmp_path / "c.yaml"
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\nbackend: hyp3\n" + extra)
    return f


def test_reference_settings_are_config_fields_with_the_same_defaults():
    import dataclasses
    dflt = {f.name: (f.default if f.default is not dataclasses.MISSING else f.default_factory())
            for f in dataclasses.fields(config.PipelineConfig) if f.name in sg.REFERENCE_SETTINGS}
    for key, (default, _meaning) in sg.REFERENCE_SETTINGS.items():
        assert key in dflt, key
        assert dflt[key] == default, key


def test_reference_settings_load_and_are_checked(tmp_path):
    poly = tmp_path / "ring.geojson"
    poly.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    cfg = load_config(str(_yaml(tmp_path, f"reference_mode: stable_polygon\nreference_stable_geojson: {poly}\n")))
    assert sg.reference_mode(cfg) == "stable_polygon"
    cfg = load_config(str(_yaml(tmp_path, "reference_mode: gnss_absolute\nreference_gnss_stations: [caha, qcap, hanf]\n"
                                          "reference_gnss_radius_m: 150\n")))
    assert sg.reference_mode(cfg) == "gnss_absolute" and cfg.reference_gnss_radius_m == 150
    with pytest.raises(ConfigError, match="at least 3"):
        load_config(str(_yaml(tmp_path, "reference_mode: gnss_absolute\nreference_gnss_stations: [CAHA, QCAP]\n")))
    with pytest.raises(ConfigError, match="at least 5"):
        load_config(str(_yaml(tmp_path, "reference_mode: gnss_absolute\nreference_gnss_plane: true\n"
                                        "reference_gnss_stations: [A, B, C, D]\n")))
    with pytest.raises(ConfigError, match="is not one of"):
        load_config(str(_yaml(tmp_path, "reference_mode: absolute\n")))
    with pytest.raises(ConfigError, match="reference_stable_geojson is empty"):
        load_config(str(_yaml(tmp_path, "reference_mode: stable_polygon\n")))
    with pytest.raises(ConfigError) as e:
        load_config(str(_yaml(tmp_path, f"reference_mode: stable_polygon\nreference_stable_geojson: {tmp_path / 'no.geojson'}\n")))
    assert str(e.value).count("does not exist") == 1                # said once, by the input-file check
    assert load_config(str(_yaml(tmp_path))).reference_mode == "farmland"


# ====================================================================== insar validate

def test_insar_validate_reaches_the_harness_from_the_dashboard_command(monkeypatch):
    from insar_pipeline import cli, dashboard
    seen = {}
    monkeypatch.setattr(cli, "validate_main", lambda argv: seen.setdefault("argv", argv) and 0)
    monkeypatch.setattr(sys, "argv", ["insar", "validate", "c.yaml", "--protocol", "p.yaml"])
    with pytest.raises(SystemExit) as e:
        dashboard.main()
    assert e.value.code == 0 and seen["argv"] == ["c.yaml", "--protocol", "p.yaml"]
    assert "insar validate <config> --protocol" in dashboard.USAGE
    assert dashboard.stage_label("validation").startswith("Validation against ground truth")


# ====================================================================== products stage -> real harness

def test_an_experiment_products_run_calls_the_real_harness(tmp_path, monkeypatch):
    H.make_product(tmp_path / "proj", H._dates(step=24))
    proto = H.b1_protocol(tmp_path)                     # its run T1 is the config file "synthetic.yaml"
    y = tmp_path / "synthetic.yaml"
    y.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-119.70, 36.19, -119.59, 36.30]\n"
                 "start_date: '2021-10-01'\nend_date: '2024-09-30'\nbackend: hyp3\npurpose: experiment\n"
                 f"site_profile: groundwater_basin\nvalidation_protocol: {proto}\n")
    cfg = load_config(str(y))
    calls = []
    monkeypatch.setattr(pipeline, "export_all", lambda c: calls.append("deliverables") or c.project_dir / "deliverables")
    monkeypatch.setattr(pipeline, "geocode_missing_layers", lambda c: {})
    fake = H.FakeGT(H.STATIONS_B)
    monkeypatch.setattr(V, "_ground_truth", lambda: fake)
    from insar_pipeline import provenance
    monkeypatch.setattr(provenance, "record_stages", lambda c: calls.append("provenance") or
                        c.project_dir / "deliverables" / "provenance.json")
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    out = tmp_path / "proj" / "validation" / "protocol_B1_T1"
    js = json.loads((out / "validation.json").read_text())
    assert js["protocol"]["run"] == "T1" and js["rows"]["rate_rmse_dd"]["grade"] == "pass"
    assert js["product"]["relative_figures"]["reference_mode"] == "farmland"
    assert (out / "validation.md").exists()
    assert progress.latest(cfg.project_dir)["validation"]["status"] == "done"
    assert calls == ["deliverables", "provenance"]
    assert fake.earthquake_steps is True                # protocol B1: "steps from steps.txt", both kinds


# ====================================================================== D1 output -> harness --up

def test_the_harness_reads_decompose_output_on_its_own_grid_and_epochs(tmp_path):
    asc_d = D.dates_every(date(2020, 1, 3), date(2021, 12, 30))
    desc_d = D.dates_every(date(2020, 1, 8), date(2021, 12, 30))
    a = D.make_run(tmp_path / "a", dates=asc_d, az=D.ASC_AZ, up=D.UP, east=D.EAST, name="a", **D.GRID)
    b = D.make_run(tmp_path / "b", dates=desc_d, az=D.DESC_AZ, up=D.UP, east=D.EAST, name="b",
                   **{**D.GRID, "dx": 0.003, "dy": -0.003, "nx": 20, "ny": 16})
    s = D.dc.decompose([D.geom(a, "asc_t9"), D.geom(b, "desc_t147")], tmp_path / "d1", ref="none")
    vfile = s["series"]["files"]["vertical"]
    g = s["grid"]                                       # a common-grid cell centre near the middle
    lon = g["west"] + (g["width"] // 2 + 0.5) * g["lon_step_deg"]
    lat = g["north"] + (g["length"] // 2 + 0.5) * g["lat_step_deg"]
    up = V.sample_up(vfile, {"PT01": {"lat": lat, "lon": lon}}, 60.0, 1)
    assert "North is not estimated" in up["north"] and "Up" in up["quantity"]
    ser = up["stations"]["PT01"]["series"]
    t = np.array([D.yrs(date(int(d[:4]), int(d[4:6]), int(d[6:]))) for d in up["dates"]])
    want = D.UP(t, lon, lat) - D.UP(t[0], lon, lat)
    ok = np.isfinite(ser)
    assert ok.sum() >= 20 and up["dates"][0] == s["series"]["base_epoch"].replace("-", "")
    assert np.max(np.abs(ser[ok] - want[ok])) < 0.5          # mm; the up rate field is linear in lon/lat


# ====================================================================== harness details

def test_steps_follow_the_protocol_text():
    for p in ("protocol_A.yaml", "protocol_B1.yaml", "protocol_B2.yaml"):
        prm = V.protocol_params(V.load_protocol(REVIEW / p))
        assert prm["earthquake_steps"] is True, p
    P = V.load_protocol(REVIEW / "protocol_B1.yaml")
    P["gnss_screening"] = {"method": "equipment steps only, as offsets; daily outliers beyond 5 robust sigma"}
    assert V.protocol_params(P)["earthquake_steps"] is False


def test_the_harness_and_ground_truth_map_protocol_roles_the_same_way():
    for p in ("protocol_A.yaml", "protocol_B1.yaml", "protocol_B2.yaml"):
        prm = V.protocol_params(V.load_protocol(REVIEW / p))
        pts = {q["id"]: q["role"] for q in gt.protocol_points(REVIEW / p)}
        for sid, st in prm["stations"].items():
            assert st["role"] == pts[sid], (p, sid)
    a = V.protocol_params(V.load_protocol(REVIEW / "protocol_A.yaml"))["stations"]
    assert a["PCTN"]["role"] == a["MENA"]["role"] == "null" and a["CA07"]["role"] == "reference"


def _datum_case(join_leave=True):
    """Six calibration stations: y_c(t) = o(t) + tilt(t) . xy_c + b_c, with a
    tilt that grows (0.5 mm/yr/km east); one station joins half way and one
    leaves half way."""
    rng = np.random.default_rng(7)
    n_t = 90
    days = np.arange(n_t) * 12 + V._day("2021-10-03")
    t = (days - days[0]) / 365.25
    xy = {c: tuple(v) for c, v in zip("ABCDEF", rng.uniform(-12, 12, (6, 2)))}
    xy["K"] = (9.0, -4.0)
    o = rng.normal(0, 20, n_t)
    tilt = np.c_[0.5 * t, -0.2 * t]
    Y = {}
    for c in "ABCDEF":
        Y[c] = o + tilt @ np.array(xy[c]) + rng.normal(0, 30)
    if join_leave:
        Y["E"][: n_t // 2] = np.nan
        Y["F"][n_t // 2:] = np.nan
    W = {c: np.ones(n_t) for c in Y}
    truth_k = o + tilt @ np.array(xy["K"])
    return Y, W, xy, days, truth_k


def test_joint_plane_from_the_b1_fit_is_exact_where_the_protocol_plane_is_biased():
    Y, W, xy, days, truth_k = _datum_case()
    cal = list("ABCDEF")
    pj, why = V._joint_plane_datum(cal, Y, W, xy, xy["K"], days)
    assert why is None
    e = pj - truth_k
    assert np.nanstd(e) < 1e-6                                   # a constant: removed with the station's offset
    _, consts, _ = V._loo_datum(cal, Y, W, len(days))
    pp = V._plane_datum(cal, Y, consts, xy, xy["K"], len(days))
    assert np.nanstd(pp - truth_k) > 0.3                         # the fixed protocol's two-stage plane
    # too few stations: no value, and why
    pj3, why3 = V._joint_plane_datum(list("ABC"), Y, W, xy, xy["K"], days)
    assert np.all(np.isnan(pj3)) and "needs 4" in why3


def test_the_harness_offset_datum_is_the_b1_fit():
    """Two implementations of protocol B1's gnss_absolute datum (the harness's
    leave-one-out _loo_datum and significance.gnss_datum_fit, which the
    deliverables use) give the same per-date offsets, up to the constant the
    choice of fixed station moves them by."""
    Y, W, xy, days, _ = _datum_case()
    cal = list("ABCDEF")
    o, consts, info = V._loo_datum(cal, Y, W, len(days))
    fit = sg.gnss_datum_fit(days / 365.25, np.array([Y[c] for c in cal]), sigma=np.ones((6, len(days))))
    d = o - fit["offset_mm"]
    assert np.isfinite(d).sum() == len(days) and np.nanstd(d) < 1e-8


def test_network_detail_reports_both_plane_sensitivities(tmp_path):
    res, out, _ = H._run_b(tmp_path)
    js = json.loads((out / "validation.json").read_text())
    st = js["detail"]["los"]["loo"]["stations"]
    assert all("plane_joint_rmse" in r for r in st.values()) and all("plane_rmse" in r for r in st.values())
    assert max(r["plane_joint_rmse"] for r in st.values()) < 3.0
    assert "never graded" in js["detail"]["los"]["loo"]["plane_note"] or "not graded" in js["detail"]["los"]["loo"]["plane_note"]
    assert "plane_joint_rmse" not in json.dumps(js["rows"])        # reported, not a graded row


def test_a_product_whose_gnss_datum_used_a_scored_station_is_warned(tmp_path):
    cfg = H.make_product(tmp_path / "proj", H._dates(step=24))
    prm = V.protocol_params(V.load_protocol(H.b1_protocol(tmp_path)))
    rec, warn = V.product_reference(cfg, prm)
    assert rec == {"reference_mode": "farmland"} and warn == []
    cfg.reference_mode, cfg.reference_gnss_stations = "gnss_absolute", ["aaaa", "zzzz", "yyyy"]
    (tmp_path / "proj" / "deliverables").mkdir()
    (tmp_path / "proj" / "deliverables" / "reference_datum.json").write_text(json.dumps(
        {"available": True, "stations": [{"id": "AAAA", "in_fit": True}, {"id": "ZZZZ", "in_fit": True},
                                         {"id": "YYYY", "in_fit": False}]}))
    rec, warn = V.product_reference(cfg, prm)
    assert rec["datum_stations"] == ["AAAA", "ZZZZ"] and rec["scored_stations_in_datum"] == ["AAAA"]
    assert len(warn) == 1 and "AAAA" in warn[0] and "unaffected" in warn[0]
    res, out, _ = H._run_b(tmp_path / "b")
    assert not [w for w in res["warnings"] if "gnss_absolute" in w]


# ====================================================================== nothing from an experiment reaches a client

def test_release_and_package_refuse_an_experiment_in_plain_words(tmp_path):
    from types import SimpleNamespace
    from insar_pipeline import package, release
    (tmp_path / "proj" / "deliverables").mkdir(parents=True)
    (tmp_path / "proj" / "deliverables" / "client_report.json").write_text(json.dumps({"areas": []}))
    exp = SimpleNamespace(project_dir=tmp_path / "proj", purpose="experiment", backend="miaplpy")
    b = release.blockers(exp)
    assert b[0]["code"] == "experiment_run" and b[0]["hard"] and "validation experiment" in b[0]["text"]
    assert release.status(exp)["final"] is False
    with pytest.raises(package.PackageError, match="validation experiment"):
        package.build(exp)
    client = SimpleNamespace(project_dir=tmp_path / "proj", backend="miaplpy")
    assert "experiment_run" not in [x["code"] for x in release.blockers(client)]
