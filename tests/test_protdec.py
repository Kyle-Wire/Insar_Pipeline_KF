"""Protocol decisions of 8 Oct 2026 (HANDOFF.md, "8 Oct: validation experiments -
phases 2 and 3", "Protocol decisions (8 Oct)"). Kyle delegated decisions 1, 2,
4, 5, 6 and 7 of that section; Claude made them for him, as dated amendments
appended to review_2026-10-08/protocol_{A,B1,B2}.yaml before any result:

  2026-10-08/p23-1  the Tahmoor West digitising sigma is, per point,
                    hypot(two-pass repeatability, half the plotted line width,
                    axis-calibration residual); the truth CSV carries every term
  2026-10-08/p23-2  the F2 coherence hole is RELATIVE to its ring (3 x 3 mean <=
                    2/pi of the ring's median, ring >= 0.4); the honest-or-flagged
                    row passes only with a selective flag (<= 20% of the stable
                    ground over the row's intervals), else 'flag not selective'
  2026-10-08/p23-4  Site 13 after its relocation is unit TW_G13b, own datum
  2026-10-08/p23-5  F4: cycle test in C-band LOS, accuracy in vertical, both
  2026-10-08/p23-6  A2 (DISP-S1): accuracy verdict only, labelled 'honesty not
                    assessable (...)'
  2026-10-08/p23-7  record only: the groups' readings where the protocol is silent

Every test here fails on the package and truth files before the change
(backups/pre_protdec_20261008, backups/pre_protdec_20261008_review): the
amendments, CSV columns, functions and outputs did not exist. Synthetic
products and FakeGT come from test_p1_harness; the synthetic F2 project from
test_p23_alias. No network; nothing under ~/insar_projects.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from insar_pipeline import aliasing as A                                              # noqa: E402
from insar_pipeline import lband                                                      # noqa: E402
from insar_pipeline import validation as V                                            # noqa: E402
import test_p1_harness as H                                                           # noqa: E402
import test_p23_alias as AL                                                           # noqa: E402
import test_p23_amend as AM                                                           # noqa: E402

PKG = Path(__file__).resolve().parents[1]
REVIEW = PKG / "review_2026-10-08"
TRUTH = REVIEW / "truth"
UNITS = TRUTH / "tahmoor_west_gnss_units.csv"
CALIB = TRUTH / "tahmoor_west_gnss_calibration.json"
README = PKG / "configs" / "experiments" / "README.md"
PROTOS = {n: REVIEW / f"protocol_{n}.yaml" for n in ("A", "B1", "B2")}
DELEGATED = "Claude for Kyle (delegated 8 Oct)"
A2_LABEL = "honesty not assessable (OPERA unwraps its own pairs; no per-pair aliasing record)"
WANT = {"A": ["2026-10-08/p23-1", "2026-10-08/p23-2", "2026-10-08/p23-4", "2026-10-08/p23-5",
              "2026-10-08/p23-6", "2026-10-08/p23-7"],
        "B1": ["2026-10-08/p23-5"], "B2": ["2026-10-08/p23-5"]}


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


# ===================================================================== the amendments

@pytest.mark.parametrize("name", ["A", "B1", "B2"])
def test_the_decisions_are_dated_amendments_appended_before_any_result(name):
    data = PROTOS[name].read_bytes()
    P = yaml.safe_load(data)
    am = P["amendments"]
    assert [a["id"] for a in am][-len(WANT[name]):] == WANT[name]
    for a in am[-len(WANT[name]):]:
        assert a["date"] == "2026-10-08" and a["before_any_result"] is True, a["id"]
        assert a["decided_by"] == DELEGATED, a["id"]
        assert a.get("reason") and a.get("decision") and isinstance(a.get("harness"), dict) and a["harness"], a["id"]
        assert "No result existed" in a["reason"] or "No result existed" in a["decision"], a["id"]
    # appended, not rewritten: the text above `amendments:` is still the text first fixed, and the README
    # records the file's new sha256
    assert V.fixed_text_status(PROTOS[name])["status"] == "unchanged"
    assert f"`protocol_{name}.yaml` | `{hashlib.sha256(data).hexdigest()}`" in README.read_text()
    assert V.fixed_hash_status(PROTOS[name])["status"] == "fixed"


def test_the_harness_keys_of_the_decisions():
    sw = V.amendment_switches(V.load_protocol(PROTOS["A"]))
    assert sw["digitised_sigma"] == "hypot_repeatability_line_calibration"
    assert sw["coherence_hole"] == "relative_to_ring"
    assert sw["coherence_hole_ratio"] == pytest.approx(2 / math.pi, abs=1e-4) == pytest.approx(A.HOLE_RATIO, abs=1e-4)
    assert sw["coherence_hole_block_look_cells"] == A.HOLE_BLOCK == A.GRADIENT_BLOCK
    assert sw["flag_selectivity_max_stable_share"] == 0.20
    assert sw["unit_splits"] == [{"id": "TW_G13", "from": "2022-03-28", "as": "TW_G13b"}]
    assert sw["accuracy_only_verdict"] == {"A2": A2_LABEL}
    assert sw["_by"]["accuracy_only_verdict"] == "2026-10-08/p23-6"
    rec = next(a for a in yaml.safe_load(PROTOS["A"].read_text())["amendments"] if a["id"] == "2026-10-08/p23-7")
    assert rec["kind"].startswith("record only") and len(rec["harness"]["readings_recorded"]) == 7
    for n in ("A", "B1", "B2"):
        s = V.amendment_switches(V.load_protocol(PROTOS[n]))
        assert s["lband_cycle_frame"] == "c_band_los_vertical_motion_assumed"
        assert s["lband_accuracy_frame"] == "vertical_each_sensor_own_incidence"
    # the 20% limit and its reason are stated (that the re-check is quoted from its evidence file is
    # checked locally: tests/test_local_protdec.py, the evidence names a client site)
    p2 = next(a for a in yaml.safe_load(PROTOS["A"].read_text())["amendments"] if a["id"] == "2026-10-08/p23-2")
    assert "(1 - f)" in p2["reason"] and "0.125" in p2["reason"] and "sinc(1/2)" in p2["reason"]


def test_handoff_states_the_decisions():
    from client_files import need
    text = need(PKG / "HANDOFF.md", "kept on the developer's machine").read_text()
    i = text.index("### 8 Oct: validation experiments - phases 2 and 3")
    sec = text[i:text.index("\n## ", i)]
    assert "Protocol decisions (8 Oct)" in sec and "p23-2" in sec and "flag not selective" in sec


# ===================================================================== decision 1: digitising sigma

def test_units_csv_carries_the_measured_sigma_terms():
    rows = _rows(UNITS)
    cal = json.loads(CALIB.read_text())
    for r in rows:
        for s, sig in (("", "sigma_up_mm"), ("_east", "sigma_east_mm"), ("_north", "sigma_north_mm")):
            rms, w, c = (float(r[f"{k}{s}_mm"]) for k in ("digitising_rms", "half_line_width", "calibration"))
            dig = math.sqrt(rms ** 2 + w ** 2 + c ** 2)
            assert float(r[f"digitising_sigma{s}_mm"]) == pytest.approx(dig, abs=0.011), (r["id"], r["date"], s)
            assert float(r[sig]) == pytest.approx(math.hypot(dig, float(r[f"gnss_sigma{s}_mm"])), abs=0.011)
            v1 = sig.replace("_mm", "_as_first_fixed_mm")
            half = float(r[f"half_gridline{s}_mm"])
            assert float(r[v1]) == pytest.approx(math.hypot(max(rms, half), float(r[f"gnss_sigma{s}_mm"])), abs=0.06)
        assert float(r["sigma_mm"]) == float(r["sigma_up_mm"])
    # the line width was measured from the figures: the stroke of the 7-day curves read, the legend's in every
    # figure, and half of it in mm on each value axis
    for uid, c in cal.items():
        for name, p in c["panels"].items():
            sg = p["sigma"]
            assert sg["avg7_paths_read"] >= 1 and sg["avg7_line_width_pt"] == pytest.approx(c["styles"]["avg7"][1], abs=0.01)
            b = p["pass_A"]["value_mm = a + b * y_pt"][1]
            assert sg["half_line_width_mm"] == pytest.approx(0.5 * sg["avg7_line_width_pt"] * abs(b), abs=0.006)
            assert sg["as_first_fixed"]["half_gridline_mm"] == p["pass_A"]["gridline_mm"] / 2.0
    up = np.array([float(r["sigma_up_mm"]) for r in rows])
    up1 = np.array([float(r["sigma_up_as_first_fixed_mm"]) for r in rows])
    assert up.max() < 5.0 and up1.min() >= 10.0                      # 2.2-4.1 mm, against the floor's 10-50 mm


def test_the_harness_takes_the_sigma_each_version_states():
    from insar_pipeline import ground_truth as gt
    pts = gt.load_points_csv(UNITS)
    n = sum(len(df) for _, df in pts)
    am, rec = V.digitised_truth_sigma(UNITS, pts, "hypot_repeatability_line_calibration")
    v1, rec1 = V.digitised_truth_sigma(UNITS, pts, None)
    assert rec["replaced"] == n and rec["kept_file_sigma"] == 0 and rec1["replaced"] == n
    rows = {(r["id"], r["date"]): r for r in _rows(UNITS)}
    for (pt, d_am), (_, d_v1) in zip(am, v1):
        for t, comp, s_am, s_v1 in zip(d_am["time"], d_am["component"], d_am["sigma_mm"], d_v1["sigma_mm"]):
            r = rows[(pt["id"], str(t.date()))]
            s = {"u": "", "e": "_east", "n": "_north"}[comp]
            want = math.sqrt(sum(float(r[f"{k}{s}_mm"]) ** 2 for k in ("digitising_rms", "half_line_width",
                                                                        "calibration", "gnss_sigma")))
            assert s_am == pytest.approx(want, abs=1e-9)
            assert s_v1 == float(r[{"u": "sigma_up", "e": "sigma_east", "n": "sigma_north"}[comp] + "_as_first_fixed_mm"])
    assert rec["up_sigma_mm"]["max"] < 5.0 and rec1["up_sigma_mm"]["min"] >= 10.0
    # a file without the terms keeps its own sigma, and the record says so
    other = Path(UNITS).with_name("x")                                     # not read: the columns are missing
    _, rec2 = V.digitised_truth_sigma(other, pts, "hypot_repeatability_line_calibration")
    assert rec2["replaced"] == 0 and "could not be read" in rec2["why_kept"]


def _points_csv(tmp_path, ids_dates, terms=(0.3, 0.25, 0.05, 3.0), file_sigma=40.0, v1_sigma=50.0):
    """A digitised-units CSV at a synthetic point (the reference station's column, as test_p1_harness's)."""
    lat, lon = H._cell_centre(36.252, H.STATIONS_A["RRRR"][1])
    p = tmp_path / "units.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lat", "lon", "date", "up_mm", "sigma_mm", "sigma_up_mm", "digitising_rms_mm",
                    "half_line_width_mm", "calibration_mm", "gnss_sigma_mm", "sigma_up_as_first_fixed_mm", "kind",
                    "role"])
        for uid, dates in ids_dates:
            days = V._days([d.replace("-", "") for d in dates])
            vals = H._vertical(lat, lon, days) + 50.0
            for d, v in zip(dates, vals):
                w.writerow([uid, lat, lon, d, round(float(v), 2), file_sigma, file_sigma, *terms, v1_sigma, "gnss",
                            "validation"])
    return p


def _run(tmp_path, scale=1.0, proto=None, outlines=None, record=None, points=None, run="T1"):
    """test_p23_amend's synthetic A run (VVVV's 300 mm event), with a per-pair aliasing record given as
    record(pair) -> (level, level_v1) full arrays, and a points CSV read by the real reader."""
    from insar_pipeline import ground_truth as gtm
    dates = H._dates()
    days = V._days(dates)
    lat, lon = H._cell_centre(*H.STATIONS_A["VVVV"])
    r, c = H._rc(lat, lon)
    inc, az = H._geometry()[:2]
    lu = H._los(inc[r, c], az[r, c])[2]
    extra = np.zeros((len(dates), H.NY, H.NX), np.float32)
    extra[:, r - 4:r + 5, c - 4:c + 5] = (scale * lu * H._event(days))[:, None, None]
    extra += np.random.default_rng(11).normal(0, 1.0, extra.shape).astype(np.float32)
    root = tmp_path / "proj"
    cfg = H.make_product(root, dates, extra=extra)
    cfg.purpose, cfg.aliasing_diagnostic = "experiment", "on"
    gt = H.FakeGT(H.STATIONS_A, overrides={"VVVV": lambda d, enu: enu + np.column_stack(
        [np.zeros(d.size), np.zeros(d.size), H._event(d)])})
    if points is not None:
        gt.points = gtm.load_points_csv(points)
    if record is not None:
        rec = root / "aliasing"
        (rec / "pairs").mkdir(parents=True)
        _, _, glat, glon = H._geometry()
        np.savez_compressed(rec / "radar_grid.npz", window=np.array([0, H.NY, 0, H.NX]), frame=np.array([H.NY, H.NX]),
                            lat=glat.astype(np.float64), lon=glon.astype(np.float64), spacing=np.array([55.0, 45.0]))
        for pair in AM._chain(dates, lambda a, b: None):
            lv, lv1 = record(pair, (r, c))
            np.savez_compressed(rec / "pairs" / f"{pair}.npz", level=lv, level_v1=lv1, key=np.array("k"),
                                hole_rule=np.array(json.dumps(A.HOLE_RULE)), stats=np.array(json.dumps({"pair": pair})))
    out = tmp_path / "out"
    res = V.validate(cfg, proto or AM._a_proto(tmp_path), out, run=run, cache_dir=tmp_path / "cache", gt=gt,
                     sensitivity=False, epoch_coverage=False, outlines=outlines,
                     points=str(points) if points is not None else None)
    return res, out


def _outline(tmp_path):
    lat, lon = H._cell_centre(*H.STATIONS_A["VVVV"])
    p = tmp_path / "outlines.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [AM._square(lat, lon, 0.003, "LW999")]}))
    return p


def test_validate_uses_the_amended_sigma_and_splits_site_13(tmp_path):
    """Decisions 1 and 4 end to end: the digitised units' truth sigma is
    the hypot of the CSV's measured terms (not the file's 40 mm or the
    first-fixed 50 mm), and TW_G13's rows from 2022-03-28 are unit TW_G13b
    with their own datum: no comparison spans the relocation."""
    pts = _points_csv(tmp_path, [("TW_G13", ["2021-12-15", "2022-01-01", "2022-04-01", "2022-06-01", "2023-01-10"]),
                                 ("TW_G07", ["2021-12-15", "2022-06-01", "2023-01-10"])])
    res, out = _run(tmp_path, points=pts, outlines=_outline(tmp_path))
    comps = [c for c in res["detail"]["comparisons"] if c.get("kind") == "point_epoch"]
    s = math.sqrt(0.3 ** 2 + 0.25 ** 2 + 0.05 ** 2 + 3.0 ** 2)
    assert comps and all(c["truth_sigma"] == pytest.approx(math.hypot(s, s), abs=0.06) for c in comps)
    g13 = [c for c in comps if c["point"] == "TW_G13"]
    g13b = [c for c in comps if c["point"] == "TW_G13b"]
    assert [c["date"] for c in g13] == ["2022-01-01"] and all(c["since"] == "2021-12-15" for c in g13)
    assert [c["date"] for c in g13b] == ["2022-06-01", "2023-01-10"] and all(c["since"] == "2022-04-01" for c in g13b)
    rules = res["truth"]["points_rules"]
    assert rules["sigma"]["amendment"] == "2026-10-08/p23-1" and rules["sigma"]["replaced"] == 8
    assert rules["unit_splits"]["amendment"] == "2026-10-08/p23-4"
    assert rules["unit_splits"]["split"] == [{"id": "TW_G13", "from": "2022-03-28", "as": "TW_G13b",
                                              "rows_before": 2, "rows_after": 3}]
    assert rules["sigma_as_first_fixed"]["replaced"] == 8                  # version_1 reads the first-fixed 50 mm
    assert "TW_G13b" in res["sampling"]
    js = json.loads((out / "validation.json").read_text())
    assert js["truth"]["points_rules"]["unit_splits"]["split"][0]["as"] == "TW_G13b"


# ===================================================================== decision 2: relative hole, selectivity

def test_the_coherence_hole_is_relative_to_its_ring():
    coh = np.full((100, 100), 0.95, np.float32)
    coh[45:55, 45:55] = 0.5                       # not below 0.4, but it kept 53% of its ring's coherence
    assert A.coherence_holes(coh, None, 40.0, 40.0)[47:53, 47:53].all()
    assert not A.coherence_holes_as_first_fixed(coh, None, 40.0, 40.0).any()
    summer = np.full((100, 100), 0.45, np.float32)
    summer[45:55, 45:55] = 0.35                   # below 0.4 (a hole as first fixed), but 78% of its ring's
    assert A.coherence_holes_as_first_fixed(summer, None, 40.0, 40.0)[47:53, 47:53].all()
    assert not A.coherence_holes(summer, None, 40.0, 40.0).any()
    low = np.full((100, 100), 0.3, np.float32)
    low[45:55, 45:55] = 0.02                      # a ring below 0.4 cannot show a hole (a scene-wide low)
    assert not A.coherence_holes(low, None, 40.0, 40.0).any()
    # the stated margin: 2/pi of the ring's median
    for v, want in ((0.8 * 2 / math.pi - 0.01, True), (0.8 * 2 / math.pi + 0.02, False)):
        c = np.full((100, 100), 0.8, np.float32)
        c[45:55, 45:55] = v
        assert bool(A.coherence_holes(c, None, 40.0, 40.0)[50, 50]) is want, v
    # the cell's coherence is its 3 x 3 look-cell mean: one speckled cell is not a hole
    one = np.full((100, 100), 0.9, np.float32)
    one[50, 50] = 0.05
    assert not A.coherence_holes(one, None, 40.0, 40.0).any() and A.coherence_holes_as_first_fixed(
        one, None, 40.0, 40.0)[50, 50]


def test_ring_median_matches_a_direct_median():
    rng = np.random.default_rng(3)
    coh = rng.uniform(0.2, 1.0, (140, 140)).astype(np.float32)
    coh[:, 70:] *= 0.6                                         # not one distribution: the ring's own median matters
    rel = rng.random((140, 140)) > 0.2
    med, cnt = A.ring_median(coh, rel, 40.0, 40.0)
    rr, cc = np.mgrid[0:140, 0:140]
    for r, c in ((70, 70), (30, 100), (110, 20), (70, 40)):
        d = np.hypot((rr - r) * 40.0, (cc - c) * 40.0)
        m = rel & (d >= 1000.0) & (d <= 3000.0)
        assert med[r, c] == pytest.approx(np.median(coh[m]), abs=0.03), (r, c)
        assert cnt[r, c] == pytest.approx(m.sum(), rel=0.15)


def test_pre_unwrap_records_the_amended_level_and_the_first_fixed_one(tmp_path):
    cfg, pairs = AL.make_project(tmp_path)
    s = A.pre_unwrap(cfg, pairs, route="miaplpy")
    assert "error" not in s and s["hole_rule"]["ratio"] == pytest.approx(2 / math.pi, abs=1e-4)
    assert s["as_first_fixed"]["flagged_share"] is not None and "v3" in s["version"]
    for p in pairs:
        with np.load(A.record_dir(cfg) / "pairs" / f"{p}.npz") as z:
            assert {"level", "level_v1", "hole_rule"} <= set(z.files)
            assert A.record_hole_rule(z)["kind"] == "relative to the ring"
            # the gradient (level 2) is the same in both; only the hole differs
            assert np.array_equal(z["level"] == 2, z["level_v1"] == 2)
    names, L, st = A.load_levels(cfg)
    _, L1, _ = A.load_levels(cfg, version="as_first_fixed")
    assert L.shape == L1.shape and all("hole_share_as_first_fixed" in x for x in st)
    with pytest.raises(ValueError):
        A.load_levels(cfg, version="other")
    # a record from before the amendment (only 'level') answers the first-fixed rule with it
    old = tmp_path / "old.npz"
    np.savez_compressed(old, level=np.ones((2, 2), np.uint8), stats=np.array("{}"))
    with np.load(old) as z:
        assert A.level_key(z, "as_first_fixed") == "level" and A.record_hole_rule(z) is None


def test_interval_flags_read_the_version_asked_for(tmp_path):
    dates = H._dates()
    cfg = H.make_product(tmp_path / "proj", dates)
    cfg.purpose, cfg.aliasing_diagnostic = "experiment", "on"
    r, c = H._rc(*H._cell_centre(*H.STATIONS_A["VVVV"]))
    rec = tmp_path / "proj" / "aliasing"
    (rec / "pairs").mkdir(parents=True)
    _, _, glat, glon = H._geometry()
    np.savez_compressed(rec / "radar_grid.npz", window=np.array([0, H.NY, 0, H.NX]), frame=np.array([H.NY, H.NX]),
                        lat=glat.astype(np.float64), lon=glon.astype(np.float64), spacing=np.array([55.0, 45.0]))
    for pair in AM._chain(dates, lambda a, b: None):
        lv, lv1 = np.zeros((H.NY, H.NX), np.uint8), np.zeros((H.NY, H.NX), np.uint8)
        lv1[r - 4:r + 5, c - 4:c + 5] = 1                     # a hole only under the rule as first fixed
        np.savez_compressed(rec / "pairs" / f"{pair}.npz", level=lv, level_v1=lv1, key=np.array("k"),
                            stats=np.array(json.dumps({"pair": pair})))
    prod = V.InsarProduct(cfg)
    rows, cols = np.mgrid[r - 3:r + 4, c - 3:c + 4]
    d0, d1 = V._day("2022-07-01"), V._day("2022-09-01")
    am = V.IntervalFlags(cfg, prod)
    v1 = V.IntervalFlags(cfg, prod, version="as_first_fixed")
    assert am.ok and v1.ok and "as amended" in am.what and "as first fixed" in v1.what
    assert am.point(rows.ravel(), cols.ravel(), d0, d1)["flagged"] is False
    assert v1.point(rows.ravel(), cols.ravel(), d0, d1)["flagged"] is True


def _event_only(pair, rc):
    """A flag on the event's cells in the pairs spanning the event, under both rules."""
    a, b = pair.split("_")
    lv = np.zeros((H.NY, H.NX), np.uint8)
    if "20220601" <= a < "20230201" or "20220601" < b <= "20230201":
        r, c = rc
        lv[r - 4:r + 5, c - 4:c + 5] = 2
    return lv, lv.copy()


def _everywhere(pair, rc):
    """A flag on every cell in every pair: it excuses the wrong event, and marks all the stable ground too."""
    lv = np.ones((H.NY, H.NX), np.uint8)
    return lv, lv.copy()


def test_a_flag_that_marks_stable_ground_is_not_selective_and_fails(tmp_path):
    res, out = _run(tmp_path, scale=0.5, outlines=_outline(tmp_path), record=_everywhere)
    uf = res["rows"]["unflagged_failure_rate"]
    assert uf["metrics"]["unflagged_failures"] == 0 and uf["rate_grade"] == "pass"     # the rate alone would pass
    assert uf["grade"] == "fail" and uf["label"] == "flag not selective"
    fs = uf["flag_selectivity"]
    assert fs["stable_flagged_share"] == 1.0 and fs["selective"] is False and fs["limit"] == 0.20
    assert fs["basis"]["interval"] == fs["comparisons"] and fs["truth_flagged_share"] == 1.0
    assert res["verdict"]["route"]["grade"] == "fail"
    # as first fixed there was no selectivity rule: the same flag passed the row
    assert res["version_1"]["rows"]["unflagged_failure_rate"]["grade"] == "pass"
    md = (out / "validation.md").read_text()
    assert "flag not selective" in md and "flag on stable ground: 100.0% (limit 20%" in md


def test_a_flag_on_the_event_only_is_selective_and_the_row_passes(tmp_path):
    res, _ = _run(tmp_path, scale=0.5, outlines=_outline(tmp_path), record=_event_only)
    uf = res["rows"]["unflagged_failure_rate"]
    assert uf["grade"] == "pass" and uf["flag_selectivity"]["stable_flagged_share"] == 0.0
    assert uf["flag_selectivity"]["selective"] is True and uf["flag_selectivity"]["stable_ground"].startswith("stable ring")
    assert res["aliasing"]["interval_flags"]["hole_rule_version"] == "amended"
    assert "hole_rule_differs_from_protocol" not in res["aliasing"]["interval_flags"]
    assert not any("coherence hole is not the one" in w for w in res["warnings"])


def test_a_record_made_with_another_hole_ratio_is_warned(tmp_path, monkeypatch):
    """(independent review, 8 Oct) The protocol states the hole's ratio and cell (harness keys
    coherence_hole_ratio, coherence_hole_block_look_cells); a per-pair record made with others is
    named in the warnings, not read silently as the amended flag."""
    monkeypatch.setattr(A, "HOLE_RULE", {**A.HOLE_RULE, "ratio": 0.5, "cell": "mean over 5 x 5 look cells"})
    res, _ = _run(tmp_path, scale=0.5, outlines=_outline(tmp_path), record=_event_only)
    off = res["aliasing"]["interval_flags"]["hole_rule_differs_from_protocol"]
    assert len(off) == 2 and "ratio 0.5" in off[0] and "5 x 5" in off[1]
    assert any("coherence hole is not the one amendment 2026-10-08/p23-2 states" in w for w in res["warnings"])


def test_without_stable_ground_the_row_cannot_pass(tmp_path):
    res, _ = _run(tmp_path, scale=0.5, record=_event_only)        # no outlines, no deramp polygon
    uf = res["rows"]["unflagged_failure_rate"]
    assert uf["metrics"]["unflagged_failures"] == 0 and uf["rate_grade"] == "pass"
    assert uf["grade"] == "not_testable" and "selectivity could not be measured" in uf["why"]
    # a failing rate still fails without it
    res2, _ = _run(tmp_path / "b", scale=0.5, record=lambda p, rc: (np.zeros((H.NY, H.NX), np.uint8),) * 2)
    assert res2["rows"]["unflagged_failure_rate"]["grade"] == "fail"
    assert res2["rows"]["unflagged_failure_rate"].get("label") != "flag not selective"


# ===================================================================== decision 5: F4's two frames

def _lband_pair(c_vals, l_vals, c_up=0.78, l_up=0.60):
    g = lband.Grid(150.6, -34.1, 0.001, -0.001, 12, 12)
    dates_c = ["20260601", "20260613", "20260625"]
    dates_l = ["20260601", "20260613", "20260625"]
    c = {"dates": dates_c, "cube": np.asarray(c_vals, float), "grid": g, "wavelength_m": 0.05546576,
         "los_up": np.full((12, 12), c_up), "los_east": np.full((12, 12), -0.6)}
    l = {"dates": dates_l, "cube": np.asarray(l_vals, float), "grid": g, "wavelength_m": 0.2379,
         "los_up": np.full((12, 12), l_up)}
    return c, l


def test_f4_reads_the_cycle_test_in_c_los_and_accuracy_in_vertical():
    up = np.stack([np.zeros((12, 12)), np.full((12, 12), -40.0), np.full((12, 12), -90.0)])   # vertical motion, mm
    c, l = _lband_pair(up * 0.78, up * 0.60)
    rec = lband.cycle_consistency(c, l)
    s = rec["summary"]
    assert s["frames"]["amendment"] == "2026-10-08/p23-5" and "C-band LOS" in s["frames"]["cycle_consistency"]
    assert "vertical" in s["frames"]["accuracy"] and "assumes vertical motion" in s["accuracy_vertical"]["label"]
    for row in rec["by_interval"]:
        assert row["frame"].startswith("C-band line of sight") and row["share_agree"] == 1.0
        assert row["accuracy_vertical"]["share_agree"] == 1.0 and row["accuracy_vertical"]["median_abs_mm"] < 1e-6
    # C-band one cycle short at a third of the cells in the second interval: a whole cycle in C-band LOS, and
    # a 27.7 / up_C (35.5 mm) vertical difference that the accuracy row reports as disagreement
    c2 = c["cube"].copy()
    c2[2, :, :4] -= 1000.0 * 0.05546576 / 2.0
    c["cube"] = c2
    rec = lband.cycle_consistency(c, l)
    second = next(r for r in rec["by_interval"] if r["dates_l"] == ["20260613", "20260625"])
    assert second["share_whole_cycle"] == pytest.approx(1 / 3, abs=0.01) and second["k_counts"] == {"-1": 48}
    acc = second["accuracy_vertical"]
    assert acc["share_agree"] == pytest.approx(2 / 3, abs=0.01)
    assert acc["rms_mm"] == pytest.approx(math.sqrt((27.73 / 0.78) ** 2 / 3), abs=0.2)


def test_vertical_accuracy_shows_horizontal_motion_the_vertical_assumption_misses():
    e = np.full((5, 5), 50.0)                                    # 50 mm east, no vertical motion
    acc = lband.vertical_accuracy(-0.6 * e, 0.6 * e, np.full((5, 5), 0.78), np.full((5, 5), 0.6), plane=False)
    assert acc["cells"] == 25 and "assumes vertical motion" in acc["label"]
    assert acc["share_agree"] == 1.0                             # a uniform motion is a datum shift (median off)
    e[:, :3] = 0.0                                               # 10 of 25 cells move east: they show
    acc = lband.vertical_accuracy(-0.6 * e, 0.6 * e, np.full((5, 5), 0.78), np.full((5, 5), 0.6))
    d = 50.0 * (0.6 / 0.78 + 1.0)                                # 88.5 mm: -30/0.78 against +30/0.6
    assert acc["share_agree"] == pytest.approx(0.6) and acc["rms_mm"] == pytest.approx(math.sqrt(10 * d * d / 25), abs=0.1)


# ===================================================================== decision 6: A2's verdict

def _a_rows(**over):
    rows = {"ca19_lw902": {"grade": "pass"}, "ca19_outage": {"grade": "pass"}, "ca19_series": {"grade": "pass"},
            "tahmoor_w_units": {"grade": "not_testable"}, "appin_peak_ratio": {"grade": "not_testable"},
            "unflagged_failure_rate": {"grade": "not_testable", "label": A2_LABEL},
            "stable_ring": {"grade": "not_computed"}}
    rows.update({k: {"grade": v} for k, v in over.items()})
    return rows


EVENTS = [{"kind": "event", "error": 5.0, "masked": False, "interval_95_mm": 40.0, "inside_interval": True},
          {"kind": "event", "error": -8.0, "masked": False, "interval_95_mm": 40.0, "inside_interval": True}]


def test_a2_gets_an_accuracy_verdict_labelled_honesty_not_assessable():
    P = V.load_protocol(PROTOS["A"])
    v = V.route_verdict(P, _a_rows(), EVENTS, run="A2")
    rt = v["route"]
    assert rt["grade"] == "pass" and rt["scope"] == "accuracy only" and rt["honesty"] == A2_LABEL
    assert rt["amendment"] == "2026-10-08/p23-6" and A2_LABEL in rt["text"] and "trustworthy" not in rt["text"]
    assert v["route_rows"] == ["ca19_lw902"]
    assert set(v["route_rows_left_out"]) == {"tahmoor_w_units", "unflagged_failure_rate"}
    assert v["route_extras"]["bias"]["grade"] == "pass" and v["route_extras"]["uncertainty_calibration"]["grade"] == "pass"
    assert V.route_verdict(P, _a_rows(ca19_lw902="fail"), EVENTS, run="A2")["route"]["grade"] == "fail"
    # A1 is not accuracy-only, and A2 under the protocol as first fixed never reached a verdict
    assert V.route_verdict(P, _a_rows(), EVENTS, run="A1")["route"]["grade"] == "not_reached"
    assert V.route_verdict(V.version_1(P), _a_rows(), EVENTS, run="A2")["route"]["grade"] == "not_reached"


def test_a2s_honest_or_flagged_row_reads_the_label_and_does_not_grade():
    P = V.load_protocol(PROTOS["A"])
    sw = V.amendment_switches(P)
    spec = P["comparisons"]["unflagged_failure_rate"]
    comps = [{"kind": "event", "truth": -300.0, "error": -150.0, "masked": False}]
    row = V._unflagged_row(spec, comps, None, {}, 0.4, (None, "old flag", False), sw=sw, run="A2")
    assert row["grade"] == "not_testable" and row["label"] == A2_LABEL and row["metrics"]["unflagged_failures"] == 1
    row1 = V._unflagged_row(spec, comps, None, {}, 0.4, (None, "old flag", False), sw=sw, run="A1")
    assert row1["grade"] == "report_only"


def test_the_accuracy_only_verdict_through_the_harness(tmp_path):
    """The wiring: validate passes the run to the verdict (the synthetic run T1 stands in for A2)."""
    def as_a2(P):
        for a in P["amendments"]:
            if (a.get("harness") or {}).get("accuracy_only_verdict"):
                a["harness"]["accuracy_only_verdict"] = {"T1": A2_LABEL}
        P["comparisons"]["tahmoor_w_units"]["runs"] = ["A1", "A3", "D1"]       # as for A2: not run on it
    res, out = _run(tmp_path, outlines=_outline(tmp_path), proto=AM._a_proto(tmp_path, as_a2))
    rt = res["verdict"]["route"]
    assert rt["scope"] == "accuracy only" and rt["honesty"] == A2_LABEL and rt["grade"] in ("pass", "marginal")
    assert res["rows"]["unflagged_failure_rate"]["label"] == A2_LABEL
    assert A2_LABEL in (out / "validation.md").read_text()
