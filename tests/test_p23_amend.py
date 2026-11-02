"""Phase 2/3, group 'amend' (8 Oct 2026, before any experiment result).

Kyle's decisions on the questions phase 1 left open, written as dated
amendments at the end of the three protocols (the fixed text above
`amendments:` is untouched), and what the harness (validation.py) does with
them:

  2026-10-08/1  protocol A: the 95% interval of a displacement change is the
                harness's, from stable ground; the route rule's
                uncertainty-calibration part is then graded (it read 'not
                stated' and could never pass);
  2026-10-08/2  protocols B1/B2: york_slope is graded on per-station
                water-year (B2: annual) changes with each interval's mean
                over the stations removed; the rate-based slope and
                rate_correlation are reported, not graded;
  2026-10-08/3  protocols B1/B2: the plane check is the joint fit, and it is
                graded (beside the offset-only rows, not in the verdict);
  2026-10-08/4  GNSS daily sigma: the larger of the robust and the source's
                formal sigma, and which one was used is recorded;
  2026-10-08/6  protocol A: a run without CA19 epochs judges the reference
                station on its own dates (recorded);
  configs       the DISP-S1 configs of A2, B2, B4 and B-2b, which the harness
                finds through the amendments.

Also the harness's reading of the C-alias outputs (the F2 flag for each
comparison's own interval, from the per-pair record; the whole-record layer
only as a fallback) and of the C-events functions (events.profile for the
Appin peak ratio, events.panel_max for the Tahmoor West peak, values_at as a
cross-check of the harness's own sampling), and that both are stub-tolerant.

Every behaviour test fails on the code before this group (no amendments,
no interval flags, no peak rows, no DISP configs). Synthetic products and
FakeGT come from test_p1_harness. No network; nothing under ~/insar_projects.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from insar_pipeline import validation as V                                            # noqa: E402
import test_p1_harness as H                                                           # noqa: E402  (synthetic product, FakeGT)

PKG = Path(__file__).resolve().parents[1]
REVIEW = PKG / "review_2026-10-08"
EXP = PKG / "configs" / "experiments"
README = EXP / "README.md"
PROTOS = {n: REVIEW / f"protocol_{n}.yaml" for n in ("A", "B1", "B2")}
FIRST_FIXED = {"A": "6f2f23accccdc5d074462fe7149b339abde9a6035e6976cc6b261d2185930a69",
               "B1": "c567492b1d47c0d4a6ba397b3bac050d8cd5817d29382105c963bae4c61d88e1",
               "B2": "89e248e17c5d6693597a319fa9345e902e9bba3f438a4d24739af8c5a5bfd25f"}
DISP = {"exp_a2_appin_disp_f39360": ("A", "A2", "exp_a1_appin_des147", "39360"),
        "exp_b2_corcoran_disp_f36542": ("B1", "B2", "exp_b1_corcoran_asc137", "36542"),
        "exp_b4_corcoran_disp_f38502": ("B1", "B4", "exp_b3_corcoran_des144", "38502"),
        "exp_b2b_houston_disp_f08882": ("B2", "B-2b", "exp_b2_houston_asc34", "08882")}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ================================================================= the protocols and the README

@pytest.mark.parametrize("name", ["A", "B1", "B2"])
def test_amendments_are_appended_and_the_fixed_text_is_unchanged(name):
    data = PROTOS[name].read_bytes()
    i = data.rfind(b"\namendments:")
    assert i > 0
    # the text above `amendments:` is the text fixed on 8 Oct, byte for byte (it then ended 'amendments: []')
    assert _sha(data[:i + 1] + b"amendments: []\n") == FIRST_FIXED[name]
    st = V.fixed_text_status(PROTOS[name])
    assert st["status"] == "unchanged" and st["sha256"] == FIRST_FIXED[name]
    # the README records the amended file's sha256 (the harness reads it 'fixed') and the first-fixed one
    assert V.fixed_hash_status(PROTOS[name])["status"] == "fixed"
    text = README.read_text()
    assert f"`protocol_{name}.yaml` | `{_sha(data)}`" in text
    assert f"protocol_{name}.yaml (first fixed) | `{FIRST_FIXED[name]}`" in text
    P = yaml.safe_load(data)
    assert P["version"] == 1 and P["status"] == "fixed" and P["fixed_on"] == "2026-10-08"
    for a in P["amendments"]:
        assert a["date"] == "2026-10-08" and a["before_any_result"] is True, a["id"]
        assert str(a["id"]).startswith("2026-10-08/") and a.get("reason") and a.get("decision"), a["id"]
        assert isinstance(a.get("harness"), dict) and a["harness"], a["id"]
        # Kyle's own decisions; the phase 2/3 decisions he delegated (8 Oct, tests/test_protdec.py); records only
        assert (a.get("decided_by") in ("Kyle", "Claude for Kyle (delegated 8 Oct)")
                or a.get("kind", "").startswith("record only")), a["id"]


def test_each_decision_is_in_the_protocols_it_applies_to():
    ids = {n: [a["id"] for a in yaml.safe_load(PROTOS[n].read_text())["amendments"]] for n in PROTOS}
    # (the finishing pass appended record-only amendments after these: the A runs' digitised deramp
    # polygon and the NISAR configs, tests/test_p23_finish.py; then the delegated phase 2/3 decisions,
    # tests/test_protdec.py)
    assert ids["A"][:6] == ["2026-10-08/1", "2026-10-08/4", "2026-10-08/6", "2026-10-08/configs",
                            "2026-10-08/a-deramp-polygon", "2026-10-08/nisar-configs"]
    assert ids["B1"][:5] == ["2026-10-08/2", "2026-10-08/3", "2026-10-08/4", "2026-10-08/configs",
                             "2026-10-08/nisar-configs"]
    assert ids["B2"] == ids["B1"]
    A, B1, B2 = (V.amendment_switches(V.load_protocol(PROTOS[n])) for n in ("A", "B1", "B2"))
    # (1) A: the interval of a change is computed from stable ground; the method is stated
    assert A["event_change_sigma"] == "stable_ground" and A["uncertainty_share_pass"] == 0.90
    method = next(a for a in yaml.safe_load(PROTOS["A"].read_text())["amendments"] if a["id"] == "2026-10-08/1")["method"]
    for words in ("1.4826 x median", "about zero", "stable ring", "deramp", "0.5-2 x", "fewer than 30", "1.96 x"):
        assert words in method, words
    # (2) the scale is graded on changes: water years at B1, annual at B2; the rate rows are reported only
    assert B1["york_slope_on"] == B2["york_slope_on"] == "interval_changes"
    assert B1["change_intervals"] == "water_years" and B2["change_intervals"] == "annual_from_window_start"
    assert B1["rate_slope_and_correlation"] == B2["rate_slope_and_correlation"] == "report_only"
    # (3) the joint plane fit is the graded plane check; (4) GNSS daily sigma; (6) A3's reference epochs
    assert B1["plane_check"] == B2["plane_check"] == "joint_graded"
    assert A["gnss_daily_sigma"] == B1["gnss_daily_sigma"] == B2["gnss_daily_sigma"] == "larger_of_robust_and_formal"
    assert A["reference_epochs_without_series_row"] == "run_dates_in_window"
    # who set what is kept, and the protocol as first fixed has none of it
    assert B1["_by"]["plane_check"] == "2026-10-08/3"
    P1 = V.version_1(V.load_protocol(PROTOS["B1"]))
    assert P1["amendments"] == [] and V.amendment_switches(P1) == {}
    assert V.protocol_params(P1, run="B1")["amend"] == {}


def test_rewritten_fixed_text_is_caught(tmp_path):
    t = PROTOS["A"].read_text().replace("coherence_floor: 0.4", "coherence_floor: 0.3", 1)
    p = tmp_path / "protocol_A.yaml"                      # the fixed name, edited ABOVE the amendments
    p.write_text(t)
    assert V.fixed_text_status(p)["status"] == "changed"
    q = tmp_path / "x" / "protocol_A.yaml"                # only the amendments edited: the fixed text still checks
    q.parent.mkdir()
    q.write_text(PROTOS["A"].read_text() + "  - id: \"2026-10-09/7\"\n    harness: {x: 1}\n")
    assert V.fixed_text_status(q)["status"] == "unchanged"


# ================================================================= the DISP-S1 configs

def test_a_run_may_be_several_configs_and_an_unlisted_config_is_said_so(tmp_path):
    P = yaml.safe_load(PROTOS["B1"].read_text())
    P["amendments"].append({"id": "2026-10-09/configs", "harness": {"run_configs": {
        "B5": ["configs/experiments/exp_b5a_x.yaml", "configs/experiments/exp_b5b_y.yaml"]}}})
    p = tmp_path / "protocol_B1x.yaml"
    p.write_text(yaml.safe_dump(P, sort_keys=False))
    Q = V.load_protocol(p)
    assert V._find_run(Q, "/any/where/exp_b5b_y.yaml") == "B5"
    assert V._find_run(Q, "exp_b2_corcoran_disp_f36542.yaml") == "B2"        # the earlier amendment still holds
    assert V._find_run(Q, "exp_unknown.yaml") is None
    # a config the protocol does not list: computed, but the output says these are no run's results
    res, _, _ = H._run_b(tmp_path)                                           # synthetic.yaml is runs.T1
    assert not any("not one of the protocol's runs" in w for w in res["warnings"])
    cfg = H.make_product(tmp_path / "p2", H._dates(step=24))
    res2 = V.validate(cfg, H.b1_protocol(tmp_path), tmp_path / "o2", cache_dir=tmp_path / "c2",
                      gt=H.FakeGT(H.STATIONS_B), sensitivity=False, epoch_coverage=False, config_path="other.yaml")
    assert any("not one of the protocol's runs" in w for w in res2["warnings"])


def test_disp_configs_are_recorded_in_the_readme():
    text = README.read_text()
    for name, (_, run, _, frame) in DISP.items():
        assert f"`{name}.yaml`" in text and f"F{frame}" in text


# ================================================================= (4) GNSS daily sigma

def _tenv3(path: Path, days, sig=(0.0010, 0.0012, 0.0045)):
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    lines = ["site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m) _north(m) u0(m) ____up(m) "
             "_ant(m) sig_e(m) sig_n(m) sig_u(m) __corr_en __corr_eu __corr_nu _latitude(deg) _longitude(deg) __height(m)"]
    for d in days:
        y, m, dd = str(np.datetime64(int(d), "D")).split("-")
        lines.append(f"TEST {y[2:]}{months[int(m) - 1]}{dd} 2022.0000 59000 2200 0 -119.6 0 0.0 0 0.0 0 0.0 0.0 "
                     f"{sig[0]} {sig[1]} {sig[2]} 0 0 0 36.25 -119.65 50.0")
    path.write_text("\n".join(lines) + "\n")


def _station(tmp_path, noise_up, formal_up):
    days = np.arange(V._day("2022-01-01"), V._day("2023-01-01"))
    rng = np.random.default_rng(4)
    enu = np.column_stack([np.zeros(days.size), np.zeros(days.size), rng.normal(0, noise_up, days.size)])
    f = tmp_path / f"T{noise_up}_{formal_up}.tenv3"
    _tenv3(f, days, sig=(0.001, 0.001, formal_up / 1000.0))
    # what ground_truth hands over: each day's sigma ALREADY the larger of formal and robust
    maxed = np.tile([1.0, 1.0, max(formal_up, noise_up)], (days.size, 1))
    return {"id": "TEST", "source": "ngl", "days": days, "enu": enu, "formal_sigma": maxed, "model": "linear",
            "hinge_days": [], "provenance": {"file": str(f)}}


def test_daily_sigma_is_the_larger_of_robust_and_the_sources_own_formal(tmp_path):
    up = np.array([0.0, 0.0, 1.0])
    amended = {"amend": {"gnss_daily_sigma": "larger_of_robust_and_formal"}, "annual": True}
    st = _station(tmp_path, noise_up=2.0, formal_up=6.0)
    r = V.daily_sigma_record(st, up, amended)
    assert abs(r["formal_mm"] - 6.0) < 1e-6 and abs(r["robust_mm"] - 2.0) < 0.3
    assert r["used"] == "formal" and r["used_mm"] == r["formal_mm"] and "tenv3" in r["formal_source"]
    assert "2026-10-08/4" in r["rule"]
    # as first fixed: the robust sigma
    r1 = V.daily_sigma_record(_station(tmp_path, 2.0, 6.0), up, {"amend": {}, "annual": True})
    assert r1["used"] == "robust" and abs(r1["used_mm"] - r1["robust_mm"]) < 1e-12
    # the formal is read from the file, not from ground_truth's per-day sigma (which is the larger already):
    # with the formal BELOW the scatter, the robust is used, and 'formal' is the file's 1 mm
    st2 = _station(tmp_path, noise_up=5.0, formal_up=1.0)
    r2 = V.daily_sigma_record(st2, up, amended)
    assert abs(r2["formal_mm"] - 1.0) < 1e-6 and r2["used"] == "robust" and r2["used_mm"] > 4.0
    # no file: ground_truth's per-day sigma stands in, labelled an upper bound
    st3 = dict(_station(tmp_path, 5.0, 1.0), provenance={})
    r3 = V.daily_sigma_record(st3, up, amended)
    assert "upper bound" in r3["formal_source"] and abs(r3["formal_mm"] - 5.0) < 1e-6
    # projected onto a LOS vector: sqrt(sum (l_i s_i)^2), median over the days
    los = np.array([0.6, -0.1, 0.79])
    los = los / np.linalg.norm(los)
    r4 = V.daily_sigma_record(_station(tmp_path, 2.0, 6.0), los, amended)
    assert abs(r4["formal_mm"] - math.sqrt((los[0] * 1) ** 2 + (los[1] * 1) ** 2 + (los[2] * 6) ** 2)) < 1e-6


# ================================================================= (2) scale graded on changes

def test_interval_changes_pick_the_dates_most_stations_share():
    sar = np.arange(V._day("2021-10-05"), V._day("2021-10-05") + 361, 30)        # 13 dates, 30 d apart
    n = sar.size
    I = {s: np.arange(n, dtype=float) for s in "PQR"}
    G = {s: 2.0 * np.arange(n, dtype=float) for s in "PQR"}
    I["R"][:2] = np.nan                                     # R starts at the third date
    sg = {s: np.ones(n) for s in "PQR"}
    prm = {"change_intervals": [("WY2022", V._day("2021-10-01"), V._day("2022-10-01"))], "change_min_years": 0.75}
    iv = V.interval_changes(sar, I, G, sg, prm, list("PQR"))[0]
    # the most stations (3) at two dates >= 0.75 yr apart: only (date 2, date 12)
    assert (iv["t0"], iv["t1"]) == (V._iso(sar[2]), V._iso(sar[12])) and len(iv["stations"]) == 3
    assert iv["stations"][0]["insar"] == 10.0 and iv["stations"][0]["gnss"] == 20.0
    assert abs(iv["stations"][0]["s_gnss"] - math.sqrt(2)) < 1e-12        # sqrt(2) x the daily sigma
    I["R"][12] = np.nan                                     # now no pair holds 3: ties at 2 go to the longest span
    iv = V.interval_changes(sar, I, G, sg, prm, list("PQR"))[0]
    assert (iv["t0"], iv["t1"]) == (V._iso(sar[0]), V._iso(sar[12])) and len(iv["stations"]) == 2
    prm["change_min_years"] = 1.5                           # nothing that far apart in one water year
    assert "span less than" in V.interval_changes(sar, I, G, sg, prm, list("PQR"))[0]["why"]


def test_change_slope_removes_each_intervals_common_motion():
    rng = np.random.default_rng(8)
    sts = [f"S{k}" for k in range(7)]
    ivs = []
    for k in range(5):
        g = rng.normal(-80, 60, len(sts))                   # truth changes, a different year each time
        ref = rng.normal(0, 200)                            # the reference pixel's own motion that year
        ivs.append({"interval": f"Y{k}", "stations": [
            {"station": s, "gnss": float(g[j]), "insar": float(0.7 * g[j] + ref + rng.normal(0, 2)), "s_gnss": 2.0}
            for j, s in enumerate(sts)]})
    out = V.change_slope(ivs, {s: 1.5 for s in sts}, 2000, 1)
    assert out["n"] == 35 and abs(out["fit"]["slope"] - 0.7) < 0.03
    assert out["ci_includes_1"] is False and out["ci_slope"]["hi"] < 0.8
    assert out["r"] > 0.99


def _b_scaled(tmp_path, scale):
    """test_p1_harness's Corcoran-like box (whole box sinks by water year), with the
    product holding `scale` x every motion."""
    dates = H._dates()
    days = V._days(dates)
    inc, az, lat, lon = H._geometry()
    L = H._los(inc, az)
    wy = {2022: -250.0, 2023: 40.0, 2024: -60.0}

    def common(d):
        d = np.asarray(d, float)
        out = np.zeros(d.size)
        for y, r in wy.items():
            a, b = V._day(f"{y - 1}-10-01"), V._day(f"{y}-10-01")
            out += r * np.clip(d - a, 0, b - a) / 365.25
        return out

    t = H._day_years(days)[:, None, None]
    rate = -40.0 + 400.0 * (lat - 36.25) - 300.0 * (lon + 119.65)
    U = rate[None] * t + 6.0 * np.sin(2 * np.pi * days / 365.25)[:, None, None]
    base = L[0][None] * H.PLATE[0] * t + L[1][None] * H.PLATE[1] * t + L[2][None] * U
    whole = base + L[2][None] * common(days)[:, None, None]
    rel = whole - whole[:, H.NY // 2, H.NX // 2][:, None, None]
    base_rel = base - base[:, H.NY // 2, H.NX // 2][:, None, None]
    extra = (scale * rel - base_rel).astype(np.float32)
    gt = H.FakeGT(H.STATIONS_B, overrides={s: (lambda d, enu: enu + np.column_stack(
        [np.zeros(d.size), np.zeros(d.size), common(d)])) for s in H.STATIONS_B})
    cfg = H.make_product(tmp_path / "proj", dates, extra=extra)
    out = tmp_path / "out"
    res = V.validate(cfg, _b1_proto(tmp_path), out, run="T1", cache_dir=tmp_path / "cache", gt=gt,
                     sensitivity=False, epoch_coverage=False, config_path=tmp_path / "synthetic.yaml")
    return res, out, gt


def _b1_proto(tmp_path) -> Path:
    """test_p1_harness's synthetic B1 protocol WITH protocol_B1.yaml's amendments (whether or not that
    helper keeps them)."""
    P = yaml.safe_load(H.b1_protocol(tmp_path).read_text())
    P["amendments"] = yaml.safe_load(PROTOS["B1"].read_text())["amendments"]
    p = tmp_path / "protocol_synthetic_b_amended.yaml"
    p.write_text(yaml.safe_dump(P, sort_keys=False))
    return p


def test_york_slope_is_graded_on_changes_and_catches_the_scale_error(tmp_path):
    res, out, _ = _b_scaled(tmp_path, 0.7)
    ys = res["rows"]["york_slope"]
    assert ys["amendment"] == "2026-10-08/2"
    # the graded slope is on the per-station water-year changes: it reads the 70% scale and fails
    assert abs(ys["metrics"]["slope"] - 0.7) < 0.05 and ys["grade"] == "fail"
    used = [iv for iv in ys["intervals"] if iv.get("t0")]
    assert ys["metrics"]["n"] >= 12 and [iv["interval"] for iv in used] == ["WY2022", "WY2023", "WY2024"]
    assert all(len(iv["stations"]) >= 2 for iv in used)
    # the rate-based slope is still computed and reported (it hides the scale error)
    assert ys["rate_based"]["slope"] > ys["metrics"]["slope"] + 0.15
    rc = res["rows"]["rate_correlation"]
    assert rc["grade"] == "report_only" and rc["amendment"] == "2026-10-08/2" and rc["metrics"]["r"] is not None
    # the protocol as first fixed: the rate-based slope graded, the correlation graded
    v1 = res["version_1"]["rows"]
    assert v1["york_slope"]["metrics"]["slope"] == pytest.approx(ys["rate_based"]["slope"])
    assert v1["rate_correlation"]["grade"] in ("pass", "marginal", "fail")
    md = (out / "validation.md").read_text()
    assert "amendment 2026-10-08/2" in md and "Under the protocol as first fixed" in md


def test_york_slope_on_changes_passes_a_right_scale_and_the_plane_check_is_graded(tmp_path):
    res, out, _ = _b_scaled(tmp_path, 1.0)
    rows = res["rows"]
    assert rows["york_slope"]["grade"] == "pass" and abs(rows["york_slope"]["metrics"]["slope"] - 1.0) < 0.03
    # (3) the joint offset + plane datum, graded against the same thresholds; not in the route verdict
    pc = rows["plane_check"]
    assert pc["amendment"] == "2026-10-08/3" and set(pc["rows"]) == {"rate_bias_loo", "series_rmse",
                                                                      "series_correlation", "uncertainty_calibration"}
    assert pc["rows"]["rate_bias_loo"]["grade"] in ("pass", "marginal", "fail")
    assert pc["rows"]["series_rmse"]["grade"] == "pass"
    assert "plane_check" not in res["verdict"]["route_rows"]
    assert res["verdict"]["route_rows"] == ["rate_rmse_dd", "rate_bias_loo", "uncertainty_calibration"]
    assert "Plane check (amendment 2026-10-08/3)" in (out / "validation.md").read_text()
    # (4) which daily sigma each station used is recorded
    ds = res["truth"]["daily_sigma"]["AAAA"]["los"]
    assert ds["used"] in ("robust", "formal") and "2026-10-08/4" in ds["rule"]
    # the protocol's own fixed-text check is in the output
    js = json.loads((out / "validation.json").read_text())
    assert js["protocol"]["amendment_keys"]["plane_check"] == "joint_graded"


def test_annual_intervals_from_the_window_start_for_b2():
    P = V.load_protocol(PROTOS["B2"])
    prm = V.protocol_params(P, run="B-2a")
    labs = [(lab, V._iso(a), V._iso(b - 1)) for lab, a, b in prm["change_intervals"]]
    assert labs == [("Y2021-10-01", "2021-10-01", "2022-09-30"), ("Y2022-10-01", "2022-10-01", "2023-09-30"),
                    ("Y2023-10-01", "2023-10-01", "2024-09-30"), ("Y2024-10-01", "2024-10-01", "2025-09-30"),
                    ("Y2025-10-01", "2025-10-01", "2026-09-30")]
    assert prm["change_min_years"] == 0.75
    p1 = V.protocol_params(V.load_protocol(PROTOS["B1"]), run="B1")
    assert [lab for lab, _, _ in p1["change_intervals"]] == [f"WY{y}" for y in range(2022, 2027)]
    # B2's DISP run ends 2026-02-19: its last year is cut at the window end
    pb = V.protocol_params(P, run="B-2b")
    assert V._iso(pb["change_intervals"][-1][2] - 1) == "2026-02-19"


# ================================================================= (1) stable-ground interval, protocol A

def test_stable_change_sigma_is_robust_about_zero_in_the_distance_band():
    rng = np.random.default_rng(3)
    d = rng.normal(4.0, 3.0, 200)                          # stable ground that shifted 4 mm: error too
    dist = np.linspace(0.5, 12.0, 200)
    sw = {"stable_distance_band": [0.5, 2.0], "stable_min_placements": 30}
    s = V.stable_change_sigma(d, dist, 4.0, sw)            # band 2-8 km
    band = (dist >= 2.0) & (dist <= 8.0)
    assert s["n"] == int(band.sum()) and s["sigma_mm"] == pytest.approx(1.4826 * np.median(np.abs(d[band])))
    assert s["sigma_mm"] > 1.4826 * np.median(np.abs(d[band] - np.median(d[band])))    # about zero, not the median
    s2 = V.stable_change_sigma(d, dist, 0.3, sw)           # band 0.15-0.6 km: too few, so all placements
    assert s2["n"] == 200 and "all placements" in s2["placements"]
    d[:185] = np.nan                                       # fewer than 30 with a value: not stated
    s3 = V.stable_change_sigma(d, dist, 4.0, sw)
    assert s3["sigma_mm"] is None and "needs 30" in s3["why"]


def _a_proto(tmp_path, mutate=None) -> Path:
    """test_p1_harness's synthetic A protocol WITH protocol_A.yaml's amendments."""
    P = yaml.safe_load(H.a_protocol(tmp_path).read_text())
    P["amendments"] = yaml.safe_load(PROTOS["A"].read_text())["amendments"]
    if mutate:
        mutate(P)
    p = tmp_path / "protocol_synthetic_a2.yaml"
    p.write_text(yaml.safe_dump(P, sort_keys=False))
    return p


def _square(lat, lon, half_deg, name="panel", **props):
    return {"type": "Feature", "properties": {"name": name, **props}, "geometry": {"type": "Polygon", "coordinates": [[
        [lon - half_deg, lat - half_deg], [lon + half_deg, lat - half_deg], [lon + half_deg, lat + half_deg],
        [lon - half_deg, lat + half_deg], [lon - half_deg, lat - half_deg]]]}}


def _run_a(tmp_path, scale=1.0, proto=None, outlines=None, lines=None, record=None, tif=None, noise=0.0, tcoh=None):
    """test_p1_harness's event at VVVV (300 mm, 2022-06 .. 2023-02), with the product holding `scale` x
    the event; `record`: per-pair aliasing levels {pair: (level at the event cells)}; `tif`: whole-record layer."""
    dates = H._dates()
    days = V._days(dates)
    lat, lon = H._cell_centre(*H.STATIONS_A["VVVV"])
    r, c = H._rc(lat, lon)
    inc, az = H._geometry()[:2]
    lu = H._los(inc[r, c], az[r, c])[2]
    extra = np.zeros((len(dates), H.NY, H.NX), np.float32)
    extra[:, r - 4:r + 5, c - 4:c + 5] = (scale * lu * H._event(days))[:, None, None]
    if noise:
        extra += np.random.default_rng(11).normal(0, noise, extra.shape).astype(np.float32)
    root = tmp_path / "proj"
    cfg = H.make_product(root, dates, extra=extra, tcoh=tcoh)
    cfg.purpose, cfg.aliasing_diagnostic = "experiment", "on"
    gt = H.FakeGT(H.STATIONS_A, overrides={"VVVV": lambda d, enu: enu + np.column_stack(
        [np.zeros(d.size), np.zeros(d.size), H._event(d)])})
    if record is not None:
        _write_record(root, dates, record, (r, c))
    if tif is not None:
        import rasterio
        (root / "deliverables").mkdir(parents=True, exist_ok=True)
        with rasterio.open(root / "deliverables" / "quality_aliasing_risk.tif", "w", driver="GTiff", height=H.NY,
                           width=H.NX, count=1, dtype="uint8", nodata=255) as dst:
            dst.write(tif.astype(np.uint8), 1)
    out = tmp_path / "out"
    res = V.validate(cfg, proto or _a_proto(tmp_path), out, run="T1", cache_dir=tmp_path / "cache", gt=gt,
                     sensitivity=False, epoch_coverage=False, outlines=outlines, lines=lines)
    return res, out, cfg


def _write_record(root: Path, dates, record: dict, rc) -> None:
    """A C-alias per-pair record on a radar grid equal to the product grid (as aliasing.pre_unwrap writes it)."""
    rec = root / "aliasing"
    (rec / "pairs").mkdir(parents=True)
    _, _, lat, lon = H._geometry()
    np.savez_compressed(rec / "radar_grid.npz", window=np.array([0, H.NY, 0, H.NX]), frame=np.array([H.NY, H.NX]),
                        lat=lat.astype(np.float64), lon=lon.astype(np.float64), spacing=np.array([55.0, 45.0]))
    r, c = rc
    for pair, lev in record.items():
        L = np.zeros((H.NY, H.NX), np.uint8)
        if lev:
            L[r - 4:r + 5, c - 4:c + 5] = lev
        np.savez_compressed(rec / "pairs" / f"{pair}.npz", level=L, key=np.array("k"),
                            stats=np.array(json.dumps({"pair": pair})))


def _chain(dates, level_for):
    """Nearest-neighbour and second-nearest pairs; level_for(a, b) -> level at the event cells."""
    out = {}
    for k in range(len(dates) - 1):
        for j in (k + 1, k + 2):
            if j < len(dates):
                out[f"{dates[k]}_{dates[j]}"] = level_for(dates[k], dates[j])
    return out


def test_route_uncertainty_part_is_graded_from_stable_ground(tmp_path):
    outl = tmp_path / "outlines.geojson"
    lat, lon = H._cell_centre(*H.STATIONS_A["VVVV"])
    outl.write_text(json.dumps({"type": "FeatureCollection", "features": [_square(lat, lon, 0.003, "LW999")]}))
    res, out, _ = _run_a(tmp_path, outlines=outl, noise=1.0)
    ev = res["detail"]["comparisons"]
    main = next(c for c in ev if c["row"] == "ev_main")
    # the event change got its 95% interval from the stable ring (1-3 km outside the outline + draw)
    assert main["interval_95_mm"] is not None and main["insar_sigma"]["n"] >= 30
    assert main["inside_interval"] is True
    sg = res["stable_ground"]
    assert "stable ring" in sg["how"] and sg["n_with_a_value"] >= 30 and sg["spacing_m"] >= 200
    uc = res["verdict"]["route_extras"]["uncertainty_calibration"]
    assert uc["amendment"] == "2026-10-08/1" and uc["grade"] == "pass" and uc["share_inside"] == 1.0
    # as first fixed it could only read 'not stated'
    v1 = res["version_1"]["verdict"]["route_extras"]["uncertainty_calibration"]
    assert v1["grade"] == "not_testable" and "not stated" in v1["why"]
    assert "route rule: uncertainty_calibration" in (out / "validation.md").read_text()


def test_a_wrong_event_falls_outside_its_stable_ground_interval(tmp_path):
    outl = tmp_path / "outlines.geojson"
    lat, lon = H._cell_centre(*H.STATIONS_A["VVVV"])
    outl.write_text(json.dumps({"type": "FeatureCollection", "features": [_square(lat, lon, 0.003, "LW999")]}))
    res, _, _ = _run_a(tmp_path, scale=0.5, outlines=outl, noise=1.0)
    main = next(c for c in res["detail"]["comparisons"] if c["row"] == "ev_main")
    assert main["inside_interval"] is False and abs(main["error"]) > 100
    assert res["verdict"]["route_extras"]["uncertainty_calibration"]["grade"] == "fail"


def test_without_outlines_or_a_deramp_polygon_the_interval_is_not_stated(tmp_path):
    res, _, _ = _run_a(tmp_path)
    uc = res["verdict"]["route_extras"]["uncertainty_calibration"]
    assert uc["grade"] == "not_testable" and "stable-ground interval" in uc["why"]
    assert any("No stable-ground interval" in w for w in res["warnings"])


def test_reference_epochs_are_the_runs_own_dates_without_a_series_row(tmp_path):
    def no_series(P):                                       # A3: no matched-series row lies in the run's dates
        P["comparisons"]["ev_series"]["epochs"] = "every T1 date in 2017-11-01 .. 2019-07-13 with a GA epoch within +-3 d"
    res, _, _ = _run_a(tmp_path, proto=_a_proto(tmp_path, no_series))
    rc = res["detail"]["reference_choice"]
    assert rc["chosen"] == "RRRR" and rc["amendment"].startswith("2026-10-08/6")
    assert "run's own dates" in rc["rule"]


# ================================================================= C-alias: the flag for each comparison's interval

def test_interval_flags_equal_aliasing_risk_between(tmp_path):
    from insar_pipeline import aliasing
    dates = H._dates()
    cfg = H.make_product(tmp_path / "proj", dates)
    cfg.purpose, cfg.aliasing_diagnostic = "experiment", "on"
    r, c = H._rc(*H._cell_centre(*H.STATIONS_A["VVVV"]))
    inside = lambda a, b: 2 if "20220601" <= a < "20230201" else (1 if a.startswith("2021") else 0)  # noqa: E731
    _write_record(tmp_path / "proj", dates, _chain(dates, inside), (r, c))
    prod = V.InsarProduct(cfg)
    fl = V.IntervalFlags(cfg, prod)
    assert fl.ok and fl.pairs == 2 * len(dates) - 3
    rows, cols = np.mgrid[r - 6:r + 7, c - 6:c + 7]
    rows, cols = rows.ravel(), cols.ravel()
    for d0, d1 in (("2022-07-01", "2022-08-15"), ("2021-10-10", "2021-12-31"), ("2023-06-01", "2024-01-01"),
                   ("2022-01-03", "2022-06-02"), ("2023-01-31", "2023-02-28")):
        want = aliasing.risk_between(cfg, d0, d1).astype(float)
        want[want == 255] = np.nan
        got = fl.levels(rows, cols, V._day(d0), V._day(d1))
        assert np.array_equal(got, want[rows, cols], equal_nan=True), (d0, d1)
    p = fl.point(rows, cols, V._day("2022-07-01"), V._day("2022-08-15"))
    assert p["flagged"] is False and 0 < p["share"] < 0.5 and p["likely_share"] == p["share"]
    rr, cc = np.mgrid[r - 3:r + 4, c - 3:c + 4]
    assert fl.point(rr.ravel(), cc.ravel(), V._day("2022-07-01"), V._day("2022-08-15"))["flagged"] is True
    assert fl.point(rr.ravel(), cc.ravel(), V._day("2023-06-01"), V._day("2024-01-01"))["flagged"] is False


def test_the_event_is_honest_when_flagged_in_its_own_interval(tmp_path):
    """F2 flags the event cells in the pairs spanning the event: the
    under-measured event (C-band missing half the bowl) is flagged for its
    interval, so it is not an unflagged failure, and the row grades on the
    interval flags (not the whole-record layer)."""
    dates = H._dates()
    lev = lambda a, b: 2 if ("20220601" <= a < "20230201" or "20220601" < b <= "20230201") else 0  # noqa: E731
    # (amendment 2026-10-08/p23-2: the row passes only with the flag shown selective on stable ground, so the
    # event's outline is given for the stable ring; tests/test_protdec.py)
    outl = tmp_path / "outlines.geojson"
    lat, lon = H._cell_centre(*H.STATIONS_A["VVVV"])
    outl.write_text(json.dumps({"type": "FeatureCollection", "features": [_square(lat, lon, 0.003, "LW999")]}))
    res, out, _ = _run_a(tmp_path, scale=0.5, record=_chain(dates, lev), outlines=outl)
    uf = res["rows"]["unflagged_failure_rate"]
    assert uf["flag_basis"]["interval"] == uf["metrics"]["n"] and "per comparison interval" in uf["flag_layer"]
    assert uf["metrics"]["unflagged_failures"] == 0 and uf["grade"] == "pass"
    main = next(c for c in res["detail"]["comparisons"] if c["row"] == "ev_main")
    assert main["flagged"] is True and main["flag"]["basis"] == "interval" and main["flag"]["likely_share"] > 0.5
    # a matched epoch's error is a residual after one offset over the whole series, so its interval is the
    # series' span: the pre-event epochs (whose residuals carry the event's error through the offset) are flagged
    early = [c for c in res["detail"]["comparisons"] if c["kind"] == "matched_epoch" and c["date"] < "20220501"]
    assert early and all(c["flagged"] is True and "series' span" in c["flag"]["interval_is"] for c in early)
    assert early[0]["flag"]["interval"][1] >= "2024-09-01"
    al = res["aliasing"]["interval_flags"]
    assert al["read"] is True and al["pairs"] == 2 * len(dates) - 3


def test_a_flag_outside_the_comparison_interval_does_not_excuse_it(tmp_path):
    """The same wrong event, flagged only in pairs from 2021 (before it). The
    whole-record layer would flag the point; the protocol's flag is per
    interval, so the event is an unflagged failure."""
    dates = H._dates()
    lev = lambda a, b: 2 if a < "20211201" else 0  # noqa: E731
    whole = np.zeros((H.NY, H.NX), np.uint8)
    r, c = H._rc(*H._cell_centre(*H.STATIONS_A["VVVV"]))
    whole[r - 4:r + 5, c - 4:c + 5] = 2
    res, _, _ = _run_a(tmp_path, scale=0.5, record=_chain(dates, lev), tif=whole)
    main = next(cc for cc in res["detail"]["comparisons"] if cc["row"] == "ev_main")
    assert main["flagged"] is False
    uf = res["rows"]["unflagged_failure_rate"]
    assert uf["metrics"]["unflagged_failures"] >= 1 and uf["by_kind"]["event"]["unflagged_failures"] == 1
    # without the per-pair record the whole-record layer stands in (it flags the point), and says so
    res2, _, _ = _run_a(tmp_path / "b", scale=0.5, tif=whole)
    main2 = next(cc for cc in res2["detail"]["comparisons"] if cc["row"] == "ev_main")
    assert main2.get("flagged") is True
    assert any("whole-record layer" in w for w in res2["warnings"])
    assert "whole-record" in res2["rows"]["unflagged_failure_rate"]["flag_basis"]["note"]


def test_aliasing_json_is_read_into_the_output(tmp_path):
    dates = H._dates()
    root = tmp_path / "proj"
    (root / "deliverables").mkdir(parents=True)
    (root / "deliverables" / "aliasing_risk.json").write_text(json.dumps({
        "version": "F2 test", "lower_bound": "a lower bound", "thresholds": {"gradient_limit_cycles_per_look_cell": 0.5},
        "pairs": [{"pair": "a"}, {"pair": "b"}], "harness": {"flagged_values": [2]}}))
    lev = lambda a, b: 1 if "20220601" <= a < "20230201" else 0      # only 'risk' (1) at the event  # noqa: E731
    res, _, _ = _run_a(tmp_path, scale=0.5, record=_chain(dates, lev))
    a = res["aliasing"]
    assert a["summary"]["version"] == "F2 test" and a["summary"]["n_pairs"] == 2 and a["summary"]["sha256"]
    assert a["interval_flags"]["flagged_values"] == [2]
    # the json says only level 2 flags: level 1 at the event does not excuse it
    main = next(c for c in res["detail"]["comparisons"] if c["row"] == "ev_main")
    assert main["flagged"] is False


# ================================================================= C-events: the peak rows

def _peak_proto(tmp_path, max_mm=200.0, tw_mm=300.0):
    def m(P):
        P["truth"]["end_of_panel_maxima"]["panels_in_window"] = {
            "LW999": {"mined": ["2022-06-01", "2023-02-01"], "max_mm": max_mm, "runs": ["T1"]},
            "LW998": {"mined": ["2023-03-01", "2023-09-01"], "max_mm": {"ARTC_rail_line": 50, "M31_west_line": 40},
                      "runs": ["T1"]},
            "LW997": {"mined": ["2022-01-01", "2022-05-01"], "max_mm": 80, "runs": ["A1"]}}
        P["truth"]["tahmoor_west_peak"]["value_mm"] = tw_mm
        P["sampling"]["lines"] = "survey lines: reliable cells within 100 m of the polyline, in 150 m bins along it"
    return _a_proto(tmp_path, m)


def _line(lat, lon0, lon1, **props):
    return {"type": "Feature", "properties": props, "geometry": {"type": "LineString",
                                                                  "coordinates": [[lon0, lat], [lon1, lat]]}}


def _peak_files(tmp_path):
    lat, lon = H._cell_centre(*H.STATIONS_A["VVVV"])
    lines = tmp_path / "lines.geojson"
    lines.write_text(json.dumps({"type": "FeatureCollection", "features": [
        _line(lat, lon - 0.008, lon + 0.008, name="LW999 XL1", panel="LW999"),
        _line(lat + 0.01, lon - 0.008, lon + 0.008, name="LW998 ARTC rail line"),     # M31 not digitised
    ]}))
    outl = tmp_path / "panels.geojson"
    outl.write_text(json.dumps({"type": "FeatureCollection", "features": [
        _square(lat, lon, 0.006, "Tahmoor West W2", start="2022-06-01", end="2022-07-01"),
        _square(lat + 0.03, lon + 0.03, 0.002, "LW999")]}))
    return lines, outl


def test_appin_peak_ratio_from_survey_lines_and_the_tahmoor_peak_from_outlines(tmp_path):
    lines, outl = _peak_files(tmp_path)
    res, out, _ = _run_a(tmp_path, scale=1.0, proto=_peak_proto(tmp_path), lines=lines, outlines=outl)
    ap = res["rows"]["appin_peak_ratio"]
    lw = next(p for p in ap["panels"] if p["panel"] == "LW999")
    # the vertical bowl (-300 mm) double-differenced to RRRR, against a 200 mm line maximum
    assert 1.3 < lw["ratio"] < 1.6 and lw["grade"] == "pass" and lw["bins_with_value"] >= 5
    assert lw["epochs"][0] <= "2022-06-01" and lw["epochs"][1] >= "2023-02-01"
    assert "assumes vertical motion" in lw["label_vertical"]
    # LW998: the ARTC line was measured (it did not move: ratio ~0, fail); the M31 line is not digitised
    artc = next(p for p in ap["panels"] if p.get("line_key") == "ARTC_rail_line")
    m31 = next(p for p in ap["panels"] if p.get("line_key") == "M31_west_line")
    assert artc["grade"] == "fail" and "names LW998 and M31" in m31["why"]
    assert not any(p["panel"] == "LW997" for p in ap["panels"])             # an A1-only panel: not this run
    # every panel-line must pass: the row reads the lowest ratio's grade, and lists what was not graded
    assert ap["grade"] == "fail" and ap["metrics"]["n"] == 2 and ap["metrics"]["n_expected"] == 3
    assert any("M31" in s for s in ap["not_graded"])
    # each line maximum joins the honest-or-flagged row
    uf = res["rows"]["unflagged_failure_rate"]
    assert uf["by_kind"]["line_max"]["n"] == 2
    # the Tahmoor West peak: panel_max over W1..W4 (its own dates ignored: the run's window), report only
    tw = res["rows"]["tahmoor_w_peak_ratio"]
    assert tw["grade"] == "report_only" and [p["panel"] for p in tw["panels"]] == ["Tahmoor West W2"]
    assert tw["panels"][0]["window"]["epoch_start"] <= "2021-10-03" and 0.6 < tw["metrics"]["ratio"] < 1.2
    # values_at agrees with the harness's own sampling at the compared points
    xc = res["events"]["values_at_crosscheck"]
    assert xc["n"] >= 2 and xc["agree"] is True
    assert "appin_peak_ratio" in (out / "validation.md").read_text()


def test_an_underestimated_bowl_is_marginal_and_an_unflagged_line_failure(tmp_path):
    lines, outl = _peak_files(tmp_path)
    res, _, _ = _run_a(tmp_path, scale=0.5, proto=_peak_proto(tmp_path), lines=lines)
    lw = next(p for p in res["rows"]["appin_peak_ratio"]["panels"] if p["panel"] == "LW999")
    assert 0.6 <= lw["ratio"] < 0.8 and lw["grade"] == "marginal"
    line = next(c for c in res["detail"]["comparisons"] if c["kind"] == "line_max" and c["point"].startswith("LW999"))
    assert line["truth"] == -200.0 and line["error"] > 40
    assert line["flag"]["basis"] == "no flag layer" and line["flagged"] is False
    assert res["rows"]["unflagged_failure_rate"]["by_kind"]["line_max"]["unflagged_failures"] >= 1
    # no outlines: the Tahmoor row says what it needs
    assert res["rows"]["tahmoor_w_peak_ratio"]["grade"] == "not_computed"
    assert "--outlines" in res["rows"]["tahmoor_w_peak_ratio"]["why"]


def test_a_deeper_bowl_than_the_line_maximum_is_not_an_error(tmp_path):
    lines, _ = _peak_files(tmp_path)
    res, _, _ = _run_a(tmp_path, scale=1.0, proto=_peak_proto(tmp_path, max_mm=150.0), lines=lines)
    line = next(c for c in res["detail"]["comparisons"] if c["kind"] == "line_max" and c["point"].startswith("LW999"))
    assert line["insar"] < line["truth"] and line["error"] == 0.0 and line["lower_bound_truth"] is True


def test_a_line_with_no_reliable_bin_is_masked_and_one_off_the_grid_is_no_comparison(tmp_path):
    lat, lon = H._cell_centre(*H.STATIONS_A["VVVV"])
    lines = tmp_path / "lines.geojson"
    lines.write_text(json.dumps({"type": "FeatureCollection", "features": [
        _line(lat, lon - 0.008, lon + 0.008, name="LW999 XL1"),
        _line(lat + 1.0, lon - 0.008, lon + 0.008, name="LW998 ARTC"),          # 110 km away: off the grid
        _line(lat + 0.01, lon - 0.008, lon + 0.008, name="LW998 M31")]}))
    r, c = H._rc(lat, lon)
    tc = np.full((H.NY, H.NX), 0.9)
    tc[r - 5:r + 6, :] = 0.2                                # the LW999 line's whole buffer below the floor
    res, _, _ = _run_a(tmp_path, scale=0.5, proto=_peak_proto(tmp_path), lines=lines, tcoh=tc)
    ap = res["rows"]["appin_peak_ratio"]
    lw = next(p for p in ap["panels"] if p["panel"] == "LW999")
    assert "masked" in lw["why"] and "grade" not in lw
    artc = next(p for p in ap["panels"] if p.get("line_key") == "ARTC_rail_line")
    assert "outside the product grid" in artc["why"]
    kinds = [c for c in res["detail"]["comparisons"] if c["kind"] == "line_max"]
    assert sorted(c["point"] for c in kinds) == ["LW998 M31_west_line", "LW999 line"]
    assert next(c for c in kinds if c["point"] == "LW999 line")["masked"] is True
    assert res["rows"]["unflagged_failure_rate"]["by_kind"]["line_max"]["masked"] == 1


def test_peak_rows_are_stub_tolerant(tmp_path, monkeypatch):
    # no lines or outlines given and none in the config: 'not computed', saying what is needed
    res, _, _ = _run_a(tmp_path, proto=_peak_proto(tmp_path))
    assert res["rows"]["appin_peak_ratio"]["grade"] == "not_computed"
    assert "--lines" in res["rows"]["appin_peak_ratio"]["why"]
    # the events module missing a function (or failing) is recorded, never raised
    from insar_pipeline import events
    lines, outl = _peak_files(tmp_path)
    monkeypatch.delattr(events, "panel_max")
    monkeypatch.setattr(events, "profile", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("half-written")))
    (tmp_path / "b").mkdir()
    res2, _, _ = _run_a(tmp_path / "b", proto=_peak_proto(tmp_path / "b"), lines=lines, outlines=outl)
    ap = res2["rows"]["appin_peak_ratio"]
    assert ap["grade"] == "not_testable" and "half-written" in ap["why"]
    assert res2["rows"]["tahmoor_w_peak_ratio"]["grade"] == "not_computed"
    assert "panel_max is not available" in res2["rows"]["tahmoor_w_peak_ratio"]["why"]
    # the config's event_lines stand in for --lines
    from insar_pipeline import validation as VV
    P = VV.load_protocol(_peak_proto(tmp_path / "b"))
    ev = VV.event_inputs(P, VV.protocol_params(P, run="T1"), SimpleNamespace(event_lines=str(lines), event_polygons=""),
                         SimpleNamespace(days=np.array([V._day("2021-10-03"), V._day("2024-09-25")])), run="T1")
    assert ev["lines_file"] == {"path": str(lines), "from": "the config's event_lines"}


def test_tahmoor_panels_are_told_from_appin_longwalls():
    for name in ("W1", "Tahmoor West W3", "TW-W4", "w2"):
        assert V.TW_PANEL_RE.search(name), name
    for name in ("LW902", "LW1", "W5", "West", "W12", "M31 west line"):
        assert not V.TW_PANEL_RE.search(name), name
    assert V._names("LW 902 crossline XL1", "LW902") and not V._names("LW9021", "LW902")
    assert V._names("LW708 - ARTC rail", "ARTC") and not V._names("LW708 M31", "ARTC")


def test_survey_lines_that_list_their_maxima_match_by_panel_and_value():
    """The T1 survey lines carry end_of_panel_maxima [{panel, max_mm}] (their names need not name the
    panel: LW902's 263 mm is on the 'ARTC (Main Southern Railway) monitoring line, Appin Area 9')."""
    a9 = {"properties": {"id": "A9", "name": "ARTC monitoring line, Area 9",
                         "end_of_panel_maxima": [{"panel": "LW902", "max_mm": 263}]}}
    a7 = {"properties": {"id": "A7", "name": "ARTC rail monitoring line, Area 7 (LW708B)",
                         "end_of_panel_maxima": [{"panel": "LW708", "max_mm": 990}]}}
    m31 = {"properties": {"id": "M31", "name": "M31 West monitoring line, Area 7 (LW708B)",
                          "end_of_panel_maxima": [{"panel": "LW708", "max_mm": 813}]}}
    plain = {"properties": {"name": "LW708 M31 west line"}}
    assert V._line_matches(a9, "LW902", None, 263) and not V._line_matches(a9, "LW903", None, 432)
    assert V._line_matches(a7, "LW708", "ARTC_rail_line", 990) and not V._line_matches(a7, "LW708", "M31_west_line", 813)
    assert V._line_matches(m31, "LW708", "M31_west_line", 813) and not V._line_matches(m31, "LW708", "ARTC_rail_line", 990)
    assert V._line_matches(plain, "LW708", "M31_west_line", 813) and not V._line_matches(plain, "LW708", "ARTC_rail_line", 990)


# ================================================================= independent review (8 Oct 2026)

def test_interval_flags_leave_out_pair_files_the_record_summary_does_not_list(tmp_path):
    """aliasing.risk_between and the whole-record tif use the pairs of the last pre-unwrap call
    (aliasing/summary.json). A pair file left in aliasing/pairs/ by an earlier network is not one of
    them, so the harness must not flag with it either."""
    from insar_pipeline import aliasing
    dates = H._dates()
    cfg = H.make_product(tmp_path / "proj", dates)
    cfg.purpose, cfg.aliasing_diagnostic = "experiment", "on"
    r, c = H._rc(*H._cell_centre(*H.STATIONS_A["VVVV"]))
    rec = _chain(dates, lambda a, b: 0)
    _write_record(tmp_path / "proj", dates, rec, (r, c))
    # a stale pair from an earlier network, likely aliased at the point, and not in summary.json
    stale = f"{dates[0]}_{dates[-1]}"
    assert stale not in rec
    L = np.zeros((H.NY, H.NX), np.uint8)
    L[r - 4:r + 5, c - 4:c + 5] = 2
    root = tmp_path / "proj" / "aliasing"
    np.savez_compressed(root / "pairs" / f"{stale}.npz", level=L, key=np.array("k"),
                        stats=np.array(json.dumps({"pair": stale})))
    (root / "summary.json").write_text(json.dumps({"pairs": [{"pair": p} for p in rec]}))
    prod = V.InsarProduct(cfg)
    fl = V.IntervalFlags(cfg, prod)
    assert fl.ok and fl.pairs == len(rec) and fl.stale_files == 1 and "left out" in fl.what
    rows, cols = np.mgrid[r - 3:r + 4, c - 3:c + 4]
    rows, cols = rows.ravel(), cols.ravel()
    want = aliasing.risk_between(cfg, "2022-07-01", "2022-08-15").astype(float)
    assert np.array_equal(fl.levels(rows, cols, V._day("2022-07-01"), V._day("2022-08-15")), want[rows, cols])
    assert fl.point(rows, cols, V._day("2022-07-01"), V._day("2022-08-15"))["flagged"] is False


def test_the_reference_area_fallback_is_said_when_the_t1_outlines_exist(tmp_path, monkeypatch):
    """Amendment 2026-10-08/1 puts the stable discs on the stable ring once the T1 outlines exist. A run
    without --outlines (or event_polygons) then falls back to the reference area, and must say so."""
    t1 = tmp_path / "t1_panels.geojson"
    lat, lon = H._cell_centre(*H.STATIONS_A["VVVV"])
    t1.write_text(json.dumps({"type": "FeatureCollection", "features": [_square(lat, lon, 0.003, "LW999")]}))
    monkeypatch.setattr(V, "T1_OUTLINES", t1)
    res, _, _ = _run_a(tmp_path / "a")
    assert any("T1 panel outlines exist" in w and "--outlines" in w for w in res["warnings"])
    res2, _, _ = _run_a(tmp_path / "b", outlines=t1, noise=1.0)
    assert not any("T1 panel outlines exist" in w for w in res2["warnings"])
    assert res2["stable_ground"]["outlines_file"]["path"] == str(t1) and res2["stable_ground"]["outlines_file"]["sha256"]
