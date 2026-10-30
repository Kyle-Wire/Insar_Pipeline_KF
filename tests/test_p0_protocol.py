"""P0.1 (review 8 Oct 2026, group 'protocol'): the validation protocols and the
experiment run configs.

What is checked, against review_2026-10-08/PLAN.md (1A, 1B, 2, 4.2, 5):
  - every Sentinel-1 run the plan names (A1 DES-147, A3 ASC-9, B1 ASC-137,
    B3 DES-144, B-2 Houston ASC-34) has a config in configs/experiments/ that
    load_config accepts, with the plan's box, window, track, direction and
    sub-swaths, project_dir $INSAR_PROJECTS/exp_<name>;
  - the phase-linked settings are the ones the Indiana record ran with, the
    GNSS tie and the frozen-MIDAS validation are off, and the plan's disk
    settings are on (A1 keeps its slcStack for A1b, and says so);
  - the deramp and reference decisions: B (whole area sinks) no deramp; A a
    linear deramp with the panel + angle-of-draw polygon (APPROXIMATE at P0.1,
    the digitised one since the phase 2/3 finishing pass), which the
    pipeline's own deramp-mask builder accepts on a uniformly coherent grid of
    the processed crop (it refuses mid-run when the fit ground does not
    surround the area), and which leaves ground > 2 km from the polygon for
    the run's reference-pixel rule;
  - the three protocol YAMLs fix AOI, window, station roles, sampling radius,
    coherence floor, reference, models and the PLAN 5.3 thresholds, name only
    existing configs, place every used station inside the processed crop, and
    their sha256 is the one recorded in configs/experiments/README.md (a
    protocol edited without recording it fails here).
None of these files existed before P0.1, so every test fails on the old tree.
No network, nothing under ~/insar_projects: INSAR_PROJECTS points at tmp_path.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

PKG = Path(__file__).resolve().parents[1]
EXP = PKG / "configs" / "experiments"
REVIEW = PKG / "review_2026-10-08"
BOX_A = [150.55, -34.30, 150.78, -34.14]
BOX_SJV = [-119.82, 36.08, -119.52, 36.35]
BOX_HOU = [-95.864, 29.651, -95.601, 29.852]
MARGIN = 0.05                                   # stack_bbox_margin_deg: the processed crop

# PLAN.md 1A / 1B / 2 / 4.2 ("stack_swaths" table) and the decisions of 8 Oct 2026
RUNS = {
    "exp_a1_appin_des147": dict(track=147, direction="descending", swaths="1 2", start="2017-11-01",
                                end="2022-12-31", bbox=BOX_A, protocol="A", run="A1"),
    "exp_a3_appin_asc9": dict(track=9, direction="ascending", swaths="1", start="2019-11-01",
                              end="2022-12-31", bbox=BOX_A, protocol="A", run="A3"),
    "exp_b1_corcoran_asc137": dict(track=137, direction="ascending", swaths="2", start="2021-10-01",
                                   end="2026-09-30", bbox=BOX_SJV, protocol="B1", run="B1"),
    "exp_b3_corcoran_des144": dict(track=144, direction="descending", swaths="2", start="2021-10-01",
                                   end="2026-09-30", bbox=BOX_SJV, protocol="B1", run="B3"),
    "exp_b2_houston_asc34": dict(track=34, direction="ascending", swaths="2", start="2021-10-01",
                                 end="2026-09-30", bbox=BOX_HOU, protocol="B2", run="B-2a"),
}
# what the Indiana phase-linked record ran with (configs/config_site_a_pl.yaml)
INDIANA_KEYS = ("stack_polarization", "stack_num_connections", "stack_num_overlap_connections",
                "max_perp_baseline", "max_temp_baseline", "stack_range_looks", "stack_azimuth_looks",
                "stack_filter_strength", "stack_unw_method", "stack_unw_full_cost", "stack_unw_defomax",
                "stack_unw_engine", "stack_unw_cost_mode", "stack_unw_max_components",
                "stack_unw_min_component_fraction", "stack_ionosphere", "stack_cleanup_burst_igrams",
                "stack_crop_unwrap", "stack_unwrap_timeout_min", "stack_use_burst2safe", "burst2safe_env",
                "stack_delete_safes_after_coreg", "stack_virtual_merge", "stack_use_gpu",
                "stack_cleanup_offsets", "stack_cleanup_misreg_inputs", "stack_cleanup_coreg_after_filter",
                "stack_cleanup_unfiltered_igrams", "stack_cleanup_burst_geometry",
                "mintpy_unwrap_error_method", "mintpy_tropo_method", "min_temporal_coherence",
                "min_spatial_coherence", "do_esd", "esd_coherence_threshold", "mintpy_closure_correction",
                "mintpy_num_workers", "miaplpy_env", "miaplpy_ministack_size", "miaplpy_range_window",
                "miaplpy_azimuth_window", "miaplpy_phase_linking_method", "miaplpy_time_lag",
                "miaplpy_network_connections", "miaplpy_pair_coherence", "miaplpy_num_processors",
                "subset_to_aoi", "stack_bbox_margin_deg")


def _protocol(name: str) -> dict:
    return yaml.safe_load((REVIEW / f"protocol_{name}.yaml").read_text())


def _inside(lat, lon, bbox, margin=0.0) -> bool:
    w, s, e, n = bbox
    return w - margin <= lon <= e + margin and s - margin <= lat <= n + margin


# ---------------------------------------------------------------- the protocols
@pytest.mark.parametrize("name", ["A", "B1", "B2"])
def test_protocol_is_fixed_and_complete(name):
    p = _protocol(name)
    assert p["protocol"] == name and p["version"] == 1 and p["status"] == "fixed"
    assert p["fixed_on"] == "2026-10-08"
    # amendments are appended, never edited in (the text above them is checked byte for byte in
    # test_p23_amend), and every one was made before any result existed
    assert all(a.get("before_any_result") is True and str(a.get("id", "")).startswith("2026-")
               for a in p["amendments"])
    for key in ("site", "window", "runs", "truth", "reference", "models", "comparisons", "verdict"):
        assert key in p, (name, key)
    base = _protocol("B1") if name == "B2" else p              # B2 inherits sampling/reliability from B1 v1
    if name == "B2":
        assert p["inherits"]["file"].endswith("protocol_B1.yaml") and p["inherits"]["version"] == 1
        assert {"sampling", "reliability", "epoch_matching", "gnss_screening"} <= set(p["inherits"]["sections"])
    assert base["sampling"]["radius_m"] == 100 and base["sampling"]["radius_m_nisar"] == 160
    assert base["sampling"]["statistic"] == "median"
    assert base["reliability"]["coherence_floor"] == 0.4
    assert "+-3 days" in base["epoch_matching"]["gnss_at_sar_date"]
    for c in p["comparisons"].values():
        g = c.get("grade") if isinstance(c, dict) else None
        if isinstance(g, dict):
            assert "pass" in g, (name, c.get("what"))


@pytest.mark.parametrize("name,box", [("A", BOX_A), ("B1", BOX_SJV), ("B2", BOX_HOU)])
def test_every_used_station_is_inside_the_processed_crop(name, box):
    p = _protocol(name)
    for sta, rec in p["truth"]["stations"].items():
        roles = rec.get("roles", rec.get("role"))
        roles = roles if isinstance(roles, list) else [roles]
        if "not_used" in roles:
            if "lat" in rec and name == "A":
                assert not _inside(rec["lat"], rec["lon"], box, MARGIN), sta    # why it is not used
            continue
        assert _inside(rec["lat"], rec["lon"], box, MARGIN), sta
    if name == "B1":
        ext = p["site"]["nisar_extension"]["bbox_wsen"]
        for sta, rec in p["truth"]["nisar_stations"]["el_nido"].items():
            if "lat" in rec:
                assert _inside(rec["lat"], rec["lon"], ext), sta        # LSJ1 too (PLAN's core box left it out)


def test_no_station_both_references_and_is_scored():
    a = _protocol("A")
    roles = {k: (v["role"] if isinstance(v["role"], list) else [v["role"]]) for k, v in a["truth"]["stations"].items()}
    assert roles["CA07"] == ["reference"] and a["reference"]["primary"]["station"] == "CA07"
    assert roles["CA19"] == ["validation"]
    assert not any("reference" in r and "validation" in r for r in roles.values())
    assert "CA19" in a["reference"]["never"]
    for name in ("B1", "B2"):
        p = _protocol(name)
        modes = p["reference"]["modes"]
        assert "double_difference" in modes and "gnss_absolute_loo" in modes
        rule = (_protocol("B1") if name == "B2" else p)["truth"]["station_rules"]["calibration_vs_validation"]
        assert "leave-one-out" in rule
    b2 = _protocol("B2")["truth"]["stations"]
    assert b2["UHKS"]["roles"] == ["duplicate_check"] and "rate" in b2["UHKD"]["roles"]   # one site, counted once


def test_thresholds_are_plan_5_3():
    a = _protocol("A")["comparisons"]
    assert a["ca19_lw902"]["grade"]["pass"] == "|err| <= max(30 mm, 0.10 x |truth|)"
    assert a["ca19_lw902"]["grade"]["marginal"] == "|err| <= 0.20 x |truth|"
    assert a["ca19_outage"]["grade"] == a["ca19_lw902"]["grade"]
    assert (a["ca19_series"]["grade"]["pass"], a["ca19_series"]["grade"]["marginal"]) == ("rmse <= 25 mm", "rmse <= 40 mm")
    assert (a["tahmoor_w_units"]["grade"]["pass"], a["tahmoor_w_units"]["grade"]["marginal"]) == ("rmse <= 30 mm", "rmse <= 60 mm")
    assert a["appin_peak_ratio"]["grade"]["pass"] == "ratio >= 0.8"
    assert a["tahmoor_w_peak_ratio"]["grade"] == "report_only"
    assert (a["unflagged_failure_rate"]["grade"]["pass"], a["unflagged_failure_rate"]["grade"]["marginal"]) == \
        ("rate <= 0.10", "rate <= 0.25")
    assert a["stable_ring"]["grade"]["pass"] == "share_beyond <= 0.05 AND rmse_vs_zero <= 5 mm/yr"
    b = _protocol("B1")["comparisons"]
    h = _protocol("B2")["comparisons"]
    for c in (b, h):
        assert c["rate_rmse_dd"]["grade"]["pass"].startswith("rmse <= 5 mm/yr")
        assert c["rate_rmse_dd"]["grade"]["marginal"] == "rmse <= 10 mm/yr"
        assert (c["rate_bias_loo"]["grade"]["pass"], c["rate_bias_loo"]["grade"]["marginal"]) == \
            ("|bias| <= 2 mm/yr", "|bias| <= 5 mm/yr")
    assert "0.95-1.05" in b["york_slope"]["grade"]["pass"] and "0.9-1.1" in h["york_slope"]["grade"]["pass"]
    assert (b["rate_correlation"]["grade"]["pass"], h["rate_correlation"]["grade"]["pass"]) == ("r >= 0.98", "r >= 0.9")
    assert b["series_rmse"]["grade"]["pass"].startswith("median rmse <= 18 mm")
    assert h["series_rmse"]["grade"]["pass"] == "median rmse <= 10 mm"
    assert b["tre_vertical"]["grade"] == {"pass": ">= 90% of cells", "marginal": ">= 75%", "fail": "< 75%"}
    assert "at most 1 outside" in b["uncertainty_calibration"]["grade"]["pass"]
    assert b["nisar"]["pair_differences"]["grade"]["pass"].startswith(">= 70%")
    # B1 is not linear: water-year hinges; Houston: linear + annual
    assert "2022-10-01" in _protocol("B1")["models"]["rate_model"]
    assert _protocol("B2")["models"]["rate_model"].startswith("linear + annual")


def test_readme_records_the_fixed_protocols_and_how_to_launch():
    text = (EXP / "README.md").read_text()
    for name in ("A", "B1", "B2"):
        digest = hashlib.sha256((REVIEW / f"protocol_{name}.yaml").read_bytes()).hexdigest()
        assert digest in text, f"protocol_{name}.yaml changed without its sha256 in README.md (amend it, record it)"
    for name in RUNS:
        assert f"insar ~/insar_pipeline/configs/experiments/{name}.yaml" in text, name
    assert "real_budget.py" in text and "--level 2" in text


def test_burst_check_record_supports_the_configs():
    """The ASF metadata check behind the configs' burst claims (skipped when the
    review scratch folder has been cleared)."""
    rec_f = REVIEW / "p0" / "protocol" / "burst_check.json"
    if not rec_f.exists():
        pytest.skip("review_2026-10-08/p0/protocol/burst_check.json not present")
    rec = json.loads(rec_f.read_text())
    for name, want in RUNS.items():
        r = rec[f"{name}.yaml"]
        touching = {k for k, v in r["burst_ids_touching_bbox_by_subswath"].items() if v}
        assert touching == {f"IW{x}" for x in want["swaths"].split()}, name     # no sub-swath dropped
        assert r["bbox_covered_by_configured_bursts"] >= 0.99, name
        assert r["reference_date_is_an_acquisition"], name
        assert all(v["in_burst_union"] for k, v in r["stations"].items() if v["in_crop"]), name
        assert r["short_dates"] == [] or name.startswith("exp_b"), name
    assert rec["exp_a1_appin_des147.yaml"]["modal_bursts_per_date"] == 4          # 2 per sub-swath, equal
    assert re.fullmatch(r"\d+", str(rec["exp_b1_corcoran_asc137.yaml"]["n_dates"]))

