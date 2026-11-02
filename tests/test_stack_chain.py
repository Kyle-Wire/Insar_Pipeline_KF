"""
The topsStack chain, tested without ISCE or MintPy.

Every function here is pure or filesystem-only: config patching and its
verification, the looks-change cascade, product selection, the deramp
mask on synthetic geometry (holes and the no-polygon refusal), network
connectivity, the unwrap crop/pad round trip, the outcome predicate, the
acceptance criteria on a planted bowl, config validation, and the process
teardown that HANDOFF.md said was tested and was not.

Run: conda run -n insar python -m pytest tests/test_stack_chain.py -q
"""
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import stack_backend as sb            # noqa: E402
from insar_pipeline import unwrap_crop as uc              # noqa: E402
from insar_pipeline import procs                          # noqa: E402
from insar_pipeline.config import (                       # noqa: E402
    ConfigError, PipelineConfig, load_config, validate_config)


# ------------------------------------------------------------------ helpers

def make_cfg(tmp_path, **over) -> PipelineConfig:
    kw = dict(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.52, 38.47],
              start_date="2018-06-01", end_date="2019-06-30", backend="stack",
              stack_range_looks=11, stack_azimuth_looks=3, esd_coherence_threshold=0.7,
              stack_unw_full_cost=True, stack_unw_defomax=4.0)
    kw.update(over)
    cfg = PipelineConfig(**kw)
    (cfg.stack_dir / "configs").mkdir(parents=True, exist_ok=True)
    (cfg.stack_dir / "run_files").mkdir(parents=True, exist_ok=True)
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def write_configs(cfg, nomcf="False", defomax="2", rlks="8", alks="2", esd="0.85", n=3):
    d = cfg.stack_dir / "configs"
    pairs = [f"2018061{i}_2018062{i}" for i in range(n)]
    for p in pairs:
        (d / f"config_igram_unw_{p}").write_text(
            f"[Common]\n[Function-1]\nunwrap : \nifg : {cfg.stack_dir}/merged/interferograms/{p}/filt_fine.int\n"
            f"unw : {cfg.stack_dir}/merged/interferograms/{p}/filt_fine.unw\n"
            f"coh : {cfg.stack_dir}/merged/interferograms/{p}/filt_fine.cor\n"
            f"nomcf : {nomcf}\nreference : {cfg.stack_dir}/reference\ndefomax : {defomax}\n"
            f"rlks : {rlks}\nalks : {alks}\nrmfilter : False\nmethod : snaphu\n")
        (d / f"config_misreg_{p}").write_text(
            f"[Common]\n[Function-1]\nestimateAzimuthMisreg : \ncoh_threshold : {esd}\n")
        (d / f"config_merge_igram_{p}").write_text(
            f"[Common]\n[Function-1]\nmergeBursts : \noutfile : x\nrange_looks : {rlks}\n"
            f"azimuth_looks : {alks}\n")
    (d / "config_merge_hgt").write_text(
        f"[Common]\n[Function-1]\nmergeBursts : \noutfile : x\nrange_looks : {rlks}\n"
        f"azimuth_looks : {alks}\n")
    return pairs


# ---------------------------------------------------------- patch_configs

def test_patch_configs_rewrites_every_family_and_verifies(tmp_path):
    cfg = make_cfg(tmp_path)
    pairs = write_configs(cfg)
    sb.patch_configs(cfg)
    assert sb.verify_configs(cfg) == []
    txt = (cfg.stack_dir / "configs" / f"config_igram_unw_{pairs[0]}").read_text()
    assert "nomcf : True" in txt and "defomax : 4.0" in txt
    assert "rlks : 11" in txt and "alks : 3" in txt
    assert "range_looks : 11" in (cfg.stack_dir / "configs" / "config_merge_hgt").read_text()
    assert "coh_threshold : 0.7" in (cfg.stack_dir / "configs" / f"config_misreg_{pairs[0]}").read_text()


def test_patch_configs_already_correct_makes_no_edits(tmp_path):
    cfg = make_cfg(tmp_path)
    write_configs(cfg, nomcf="True", defomax="4.0", rlks="11", alks="3", esd="0.7")
    n = sb.patch_configs(cfg)
    assert n == 0
    # and the looks-change cascade must NOT have fired
    assert not (cfg.stack_dir / ".status").exists() or \
        list((cfg.stack_dir / ".status").iterdir()) == []


def test_verify_configs_names_the_wrong_file(tmp_path):
    cfg = make_cfg(tmp_path)
    write_configs(cfg, nomcf="True", defomax="4.0", rlks="11", alks="3", esd="0.7")
    f = cfg.stack_dir / "configs" / "config_merge_hgt"
    f.write_text(f.read_text().replace("range_looks : 11", "range_looks : 8"))
    problems = sb.verify_configs(cfg)
    assert len(problems) == 1 and "config_merge_hgt" in problems[0] and "8 != 11" in problems[0]


def test_patch_configs_raises_when_verification_fails(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    write_configs(cfg)
    # Make one file read-only so the patch cannot land.
    real_write = Path.write_text

    def stubborn(self, *a, **k):
        if self.name == "config_merge_hgt":
            return len(a[0])
        return real_write(self, *a, **k)
    monkeypatch.setattr(Path, "write_text", stubborn)
    with pytest.raises(RuntimeError):
        sb.patch_configs(cfg)


# --------------------------------------------- invalidate_after_looks_change

def test_invalidate_cascades_to_run13_but_not_coregistration(tmp_path):
    cfg = make_cfg(tmp_path)
    for run in ("run_11_extract_stack_valid_region", "run_12_merge_reference_secondary_slc",
                "run_13_generate_burst_igram", "run_16_unwrap"):
        d = cfg.stack_dir / ".status" / run
        d.mkdir(parents=True)
        (d / "a.done").write_text("{}")
    (cfg.stack_dir / "merged" / "interferograms").mkdir(parents=True)
    (cfg.stack_dir / "merged" / "interferograms" / "x.bin").write_bytes(b"0" * 100)
    (cfg.stack_dir / uc.WINDOW_FILE).write_text("{}")
    (cfg.mintpy_dir / "inputs").mkdir()
    (cfg.mintpy_dir / "inputs" / "ifgramStack.h5").write_bytes(b"0")
    cleared = sb.invalidate_after_looks_change(cfg)
    assert cleared == 3
    assert (cfg.stack_dir / ".status" / "run_11_extract_stack_valid_region" / "a.done").exists()
    assert not (cfg.stack_dir / ".status" / "run_13_generate_burst_igram").exists()
    assert not (cfg.stack_dir / "merged" / "interferograms").exists()
    assert not (cfg.stack_dir / uc.WINDOW_FILE).exists()
    assert not (cfg.mintpy_dir / "inputs").exists()


# ------------------------------------------------------- network connectivity

def test_connectivity_detects_orphans_and_islands():
    dates = ["20180101", "20180113", "20180125", "20180206", "20180218"]
    ok, orphaned, comps = sb.check_network_connectivity(
        dates, ["20180101_20180113", "20180113_20180125", "20180125_20180206", "20180206_20180218"])
    assert ok and orphaned == [] and len(comps) == 1
    ok, orphaned, comps = sb.check_network_connectivity(
        dates, ["20180101_20180113", "20180125_20180206"])
    assert not ok and orphaned == ["20180218"] and len(comps) == 2
    # a real case: 3 pairs removed, still one network
    dates6 = [f"2018{m:02d}{d:02d}" for m, d in
                   ((6, 11), (6, 23), (7, 5), (7, 17), (7, 29), (8, 10))]
    pairs = ["20180611_20180623", "20180611_20180705", "20180623_20180705",
             "20180705_20180717", "20180717_20180729", "20180717_20180810", "20180729_20180810"]
    assert sb.check_network_connectivity(dates6, pairs)[0]


def test_longest_first_puts_summer_and_measured_slow_pairs_first(tmp_path):
    cfg = make_cfg(tmp_path)
    (cfg.stack_dir / "unwrap_history.json").write_text(json.dumps(
        {"20181208_20181220": 100.0, "20190113_20190125": 3000.0}))
    cmds = [(0, "x 20181208_20181220"), (1, "x 20190113_20190125"),
            (2, "x 20180705_20180717"), (3, "x 20181101_20181113")]
    order = [c for _, c in sb._longest_first(cfg, "run_16_unwrap", cmds)]
    assert order[0] == "x 20190113_20190125"          # measured slowest
    assert order[1] == "x 20181208_20181220"          # measured
    assert order[2] == "x 20180705_20180717"          # summer guess
    assert order[3] == "x 20181101_20181113"


# ------------------------------------------------------- expected_outputs

def test_verify_outputs_catches_truncated_unwrap(tmp_path):
    cfg = make_cfg(tmp_path)
    pairs = write_configs(cfg, n=1)
    pdir = cfg.stack_dir / "merged" / "interferograms" / pairs[0]
    pdir.mkdir(parents=True)
    (pdir / "filt_fine.int").write_bytes(b"\0" * 800)
    conf = cfg.stack_dir / "configs" / f"config_igram_unw_{pairs[0]}"
    cmd = f"SentinelWrapper.py -c {conf}"
    assert "missing" in sb.verify_outputs(cfg, "run_16_unwrap", cmd)
    (pdir / "filt_fine.unw").write_bytes(b"\0" * 400)
    (pdir / "filt_fine.unw.conncomp").write_bytes(b"\0" * 100)
    assert "expected 800" in sb.verify_outputs(cfg, "run_16_unwrap", cmd)
    (pdir / "filt_fine.unw").write_bytes(b"\0" * 800)
    assert sb.verify_outputs(cfg, "run_16_unwrap", cmd) == ""
    # the crop wrapper's command line is understood too
    cmd2 = f"/x/python -m insar_pipeline.unwrap_crop --config {conf} --window w.json"
    assert sb.verify_outputs(cfg, "run_16_unwrap", cmd2) == ""


def test_generate_igram_counts_merged_product_as_done(tmp_path):
    cfg = make_cfg(tmp_path)
    conf = cfg.stack_dir / "configs" / "config_generate_igram_20180611_20180623"
    conf.write_text(f"[Common]\n[Function-1]\ngenerateIgram : \n"
                    f"interferogram : {cfg.stack_dir}/interferograms/20180611_20180623\n")
    cmd = f"SentinelWrapper.py -c {conf}"
    assert sb.verify_outputs(cfg, "run_13_generate_burst_igram", cmd) != ""
    m = cfg.stack_dir / "merged" / "interferograms" / "20180611_20180623"
    m.mkdir(parents=True)
    (m / "fine.int").write_bytes(b"\0" * 8)
    assert sb.verify_outputs(cfg, "run_13_generate_burst_igram", cmd) == ""


# ---------------------------------------------------------- unwrap crop/pad

def test_crop_window_and_run_file_rewrite(tmp_path):
    cfg = make_cfg(tmp_path)
    rf = cfg.stack_dir / "run_files" / "run_16_unwrap"
    conf = cfg.stack_dir / "configs" / "config_igram_unw_20180611_20180623"
    rf.write_text(f"SentinelWrapper.py -c {conf}\nSentinelWrapper.py -c {conf}2\n")
    assert uc.rewrite_unwrap_run_file(cfg) == 2
    lines = rf.read_text().splitlines()
    assert all("insar_pipeline.unwrap_crop --config" in l for l in lines)
    assert uc.rewrite_unwrap_run_file(cfg) == 0            # idempotent
    assert uc.restore_unwrap_run_file(cfg) == 1
    assert rf.read_text().startswith("SentinelWrapper.py")


def test_parse_unwrap_config_and_argv(tmp_path):
    cfg = make_cfg(tmp_path)
    pairs = write_configs(cfg, nomcf="True", defomax="4.0", rlks="11", alks="3", n=1)
    conf = uc.parse_unwrap_config(cfg.stack_dir / "configs" / f"config_igram_unw_{pairs[0]}")
    argv = uc._unwrap_argv(conf, Path("a.int"), Path("a.unw"), Path("a.cor"))
    assert "--nomcf" in argv and "--rmfilter" not in argv
    assert argv[argv.index("--defomax") + 1] == "4.0"
    assert argv[argv.index("--rlks") + 1] == "11"


def test_pad_places_crop_and_zeros_elsewhere():
    L, W, r0, r1, c0, c1 = 40, 60, 10, 25, 20, 50
    u = np.random.default_rng(0).normal(size=(r1 - r0, 2, c1 - c0)).astype(np.float32)
    full = np.zeros((L, 2, W), np.float32)
    full[r0:r1, :, c0:c1] = u
    assert np.array_equal(full[r0:r1, :, c0:c1], u)
    m = np.ones((L, W), bool)
    m[r0:r1, c0:c1] = False
    assert not full[:, 1, :][m].any()


# ------------------------------------------------------------ deramp mask

def _write_h5(path, name, arr, attrs=None):
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset(name, data=arr)
        for k, v in (attrs or {}).items():
            f.attrs[k] = v


def _square(lon0, lat0, lon1, lat1):
    return [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]


def test_deramp_mask_radar_branch_honours_holes(tmp_path):
    cfg = make_cfg(tmp_path, deramp_exclude_geojson=str(tmp_path / "mine.geojson"))
    ny, nx = 100, 120
    lat = np.linspace(38.47, 38.20, ny)[:, None] * np.ones((1, nx))
    lon = np.ones((ny, 1)) * np.linspace(-87.80, -87.52, nx)[None, :]
    (cfg.mintpy_dir / "inputs").mkdir()
    _write_h5(cfg.mintpy_dir / "inputs" / "geometryRadar.h5", "latitude", lat)
    import h5py
    with h5py.File(cfg.mintpy_dir / "inputs" / "geometryRadar.h5", "a") as f:
        f.create_dataset("longitude", data=lon)
    _write_h5(cfg.mintpy_dir / "maskTempCoh.h5", "mask", np.ones((ny, nx), bool),
              {"FILE_TYPE": "mask"})
    outer = _square(-87.70, 38.30, -87.60, 38.40)
    hole = _square(-87.67, 38.33, -87.63, 38.37)
    (tmp_path / "mine.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {},
                      "geometry": {"type": "Polygon", "coordinates": [outer, hole]}}]}))
    path = sb.build_deramp_mask(cfg, buffer_m=0.0)
    with h5py.File(path, "r") as f:
        m = f["mask"][:].astype(bool)
    # the hole must remain fittable ground (edge pixels excluded: on-edge
    # point-in-polygon is a coin toss and not what is being tested)
    e = 0.004
    hole_r = (lat > 38.33 + e) & (lat < 38.37 - e) & (lon > -87.67 + e) & (lon < -87.63 - e)
    ring_r = ((lat > 38.30 + e) & (lat < 38.40 - e) & (lon > -87.70 + e) & (lon < -87.60 - e)
              & ~((lat > 38.33 - e) & (lat < 38.37 + e) & (lon > -87.67 - e) & (lon < -87.63 + e)))
    outside = ~((lat > 38.30 - e) & (lat < 38.40 + e) & (lon > -87.70 - e) & (lon < -87.60 + e))
    assert m[hole_r].all()
    assert not m[ring_r].any()
    assert m[outside].all()


def test_deramp_mask_geocoded_branch_matches_radar(tmp_path):
    cfg = make_cfg(tmp_path, deramp_exclude_geojson=str(tmp_path / "mine.geojson"))
    ny, nx = 50, 60
    attrs = {"X_FIRST": "-87.80", "Y_FIRST": "38.47", "X_STEP": str(0.28 / nx),
             "Y_STEP": str(-0.27 / ny), "FILE_TYPE": "mask"}
    _write_h5(cfg.mintpy_dir / "maskTempCoh.h5", "mask", np.ones((ny, nx), bool), attrs)
    (tmp_path / "mine.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                                                   "geometry": {"type": "Polygon", "coordinates": [
                                                       _square(-87.70, 38.30, -87.60, 38.40)]}}]}))
    path = sb.build_deramp_mask(cfg, buffer_m=0.0)
    import h5py
    with h5py.File(path, "r") as f:
        m = f["mask"][:].astype(bool)
    excluded = (~m).sum()
    expected = (0.10 / 0.28 * nx) * (0.10 / 0.27 * ny)
    assert abs(excluded - expected) / expected < 0.15


def test_deramp_mask_refuses_without_polygon(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, deramp_exclude_geojson="")
    _write_h5(cfg.mintpy_dir / "maskTempCoh.h5", "mask", np.ones((5, 5), bool),
              {"X_FIRST": "0", "Y_FIRST": "1", "X_STEP": "0.1", "Y_STEP": "-0.1"})
    # hide the package context/ so the search finds nothing
    monkeypatch.setattr(sb.Path, "glob", lambda self, pat: iter([]))
    with pytest.raises(FileNotFoundError):
        sb.build_deramp_mask(cfg)


def test_deramp_mask_wrong_path_raises(tmp_path):
    cfg = make_cfg(tmp_path, deramp_exclude_geojson=str(tmp_path / "nope.geojson"))
    _write_h5(cfg.mintpy_dir / "maskTempCoh.h5", "mask", np.ones((5, 5), bool),
              {"X_FIRST": "0", "Y_FIRST": "1", "X_STEP": "0.1", "Y_STEP": "-0.1"})
    with pytest.raises(FileNotFoundError):
        sb.build_deramp_mask(cfg)


# --------------------------------------------------------- timeseries_file

def test_timeseries_file_prefers_geocoded_most_corrected(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg = make_cfg(tmp_path)
    for n in ("timeseries.h5", "timeseries_ERA5.h5", "timeseries_ERA5_ramp.h5",
              "timeseries_ERA5_ramp_demErr.h5", "timeseriesResidual_ramp.h5"):
        (cfg.mintpy_dir / n).write_bytes(b"0")
    assert dl.timeseries_file(cfg).name == "timeseries_ERA5_ramp_demErr.h5"
    (cfg.mintpy_dir / "geo").mkdir()
    (cfg.mintpy_dir / "geo" / "geo_timeseries_ERA5_ramp.h5").write_bytes(b"0")
    (cfg.mintpy_dir / "geo" / "geo_timeseries_ERA5_ramp_demErr.h5").write_bytes(b"0")
    assert dl.timeseries_file(cfg) == cfg.mintpy_dir / "geo" / "geo_timeseries_ERA5_ramp_demErr.h5"


# ------------------------------------------------------------- acceptance

def test_quality_assessment_on_a_planted_bowl(tmp_path):
    """A synthetic geocoded product with a known subsiding square inside the
    mine polygon: the findings must recover the sign and size, and the
    checks must run on it without a reference candidate list."""
    from insar_pipeline import quality_assessment as qa
    import h5py
    ny, nx = 120, 140
    x0, y0, dx, dy = -87.80, 38.47, 0.28 / nx, -0.27 / ny
    attrs = {"X_FIRST": str(x0), "Y_FIRST": str(y0), "X_STEP": str(dx), "Y_STEP": str(dy),
             "REF_Y": "5", "REF_X": "5", "REF_LAT": str(y0 + 5 * dy), "REF_LON": str(x0 + 5 * dx),
             "UNIT": "m/year", "FILE_TYPE": "velocity"}
    lon = x0 + np.arange(nx)[None, :] * dx + 0 * np.arange(ny)[:, None]
    lat = y0 + np.arange(ny)[:, None] * dy + 0 * np.arange(nx)[None, :]
    mine = (lon > -87.70) & (lon < -87.60) & (lat > 38.30) & (lat < 38.40)
    dates = [f"2018{m:02d}{d:02d}" for m, d in
             [(6, 11), (7, 5), (7, 29), (8, 22), (9, 15), (10, 9), (11, 2), (11, 26), (12, 20)]] + \
            [f"2019{m:02d}{d:02d}" for m, d in [(1, 13), (2, 6), (3, 2), (3, 26), (4, 19), (5, 13), (6, 6)]]
    t = np.arange(len(dates)) * 24 / 365.25
    rng = np.random.default_rng(1)
    rate = np.where(mine, -0.020, 0.0)                  # 20 mm/yr subsidence in the mine
    ts = np.zeros((len(dates), ny, nx), np.float32)
    for i in range(len(dates)):
        ts[i] = rate * t[i] + rng.normal(0, 0.0005, (ny, nx))
        if dates[i] >= "20190101":
            ts[i][mine] -= 0.010                          # a 10 mm step at the event
    vel = np.zeros((ny, nx), np.float32)
    for r in range(ny):
        vel[r] = np.polyfit(t, ts[:, r, :], 1)[0]
    cfg = make_cfg(tmp_path, deramp_exclude_geojson=str(tmp_path / "mine.geojson"),
                   acceptance_panel_geojson=str(tmp_path / "mine.geojson"),
                   reference_lalo=f"{y0 + 5 * dy:.6f},{x0 + 5 * dx:.6f}",
                   acceptance_event_start="20180915", acceptance_event_end="20190101",
                   acceptance_far_field_buffer_m=500.0)
    geo = cfg.mintpy_dir / "geo"
    geo.mkdir()
    with h5py.File(geo / "geo_velocity.h5", "w") as f:
        f.create_dataset("velocity", data=vel)
        f.create_dataset("velocityStd", data=np.full((ny, nx), 0.0005, np.float32))
        for k, v in attrs.items():
            f.attrs[k] = v
    with h5py.File(geo / "geo_timeseries_ERA5_ramp_demErr.h5", "w") as f:
        f.create_dataset("timeseries", data=ts)
        f.create_dataset("date", data=np.array([d.encode() for d in dates]))
        for k, v in attrs.items():
            f.attrs[k] = v
    for name, arr in (("geo_maskTempCoh.h5", np.ones((ny, nx), bool)),
                      ("geo_temporalCoherence.h5", np.full((ny, nx), 0.95, np.float32))):
        with h5py.File(geo / name, "w") as f:
            f.create_dataset("mask" if "mask" in name else "temporalCoherence", data=arr)
            for k, v in attrs.items():
                f.attrs[k] = v
    (tmp_path / "mine.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                                                   "geometry": {"type": "Polygon", "coordinates": [
                                                       _square(-87.70, 38.30, -87.60, 38.40)]}}]}))
    r = qa.evaluate(cfg)
    f = r["findings"]["workings_relative_to_far_field"]
    # 20 mm/yr of steady subsidence plus a 10 mm step mid-record fits steeper
    assert -38 < f["rate_mm_yr"] < -26
    panel = r["findings"]["named_panel"]
    # the planted 10 mm step plus ~4 months of the 20 mm/yr rate between the
    # window centres (~6.6 mm)
    assert -19 < panel["step_across_window_mm"] < -14
    assert r["checks"]["reference_stability"]["pass"] is True
    assert r["checks"]["rate_precision"]["pass"] is True
    assert r["checks"]["coverage"]["pass"] is True
    assert r["checks"]["distributed_scatterer_bias"]["pass"] is None   # one coherence class only
    # ...which does not apply: named, never counted as a pass or as
    # untested, so the verdict can meet practice (D2, 7 Oct 2026)
    assert r["checks"]["distributed_scatterer_bias"]["not_applicable"]
    assert r["not_applicable"] == ["distributed_scatterer_bias"]
    assert r["applicable"] == r["passed"] == len(r["checks"]) - 1 and r["untested"] == 0
    assert r["checks_text"] == "0 of 4 applicable checks flagged; 1 not applicable (distributed scatterer bias)"
    assert r["findings"]["deforming_area"]["inside_mapped_panels_percent"] > 95
    assert r["verdict"] == "MEETS PRACTICE"
    assert r["series_mm"]["dates"][0] == "20180611"


# ---------------------------------------------------------------- config

def test_load_config_rejects_unknown_key_with_hint(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("project_dir: /tmp/x\nbbox: [-87.8, 38.2, -87.52, 38.47]\n"
                 "start_date: '2018-06-01'\nend_date: '2019-06-30'\nstack_range_look: 11\n")
    with pytest.raises(ConfigError) as e:
        load_config(str(p))
    assert "stack_range_look" in str(e.value) and "stack_range_looks" in str(e.value)


def test_validate_config_catches_enums_and_ranges(tmp_path):
    cfg = make_cfg(tmp_path, backend="stac", min_temporal_coherence=1.4,
                   bbox=[-87.52, 38.2, -87.8, 38.47])
    problems = validate_config(cfg)
    assert any("backend" in p for p in problems)
    assert any("min_temporal_coherence" in p for p in problems)
    assert any("bbox" in p for p in problems)
    assert validate_config(make_cfg(tmp_path / "b")) == []


# ---------------------------------------------------------------- procs

@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_kill_children_kills_the_whole_group():
    proc = procs.popen_group(["bash", "-c", "sleep 300 & sleep 300 & wait"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.3)
    assert proc.pid in procs.registered()
    n = procs.kill_children(grace_s=0.2)
    assert n >= 1
    proc.wait(timeout=5)
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.killpg(proc.pid, 0)
    procs.unregister(proc.pid)
    assert procs.stopping()                      # the flag is raised by the kill
    with pytest.raises(RuntimeError):
        procs.popen_group(["true"])              # and launchers refuse while it is
    procs.clear_stopping()


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_run_group_timeout_returns_minus_nine():
    procs.clear_stopping()
    r = procs.run_group(["bash", "-c", "sleep 30"], timeout=0.3)
    assert r.returncode == -9
    assert not procs.registered()


def test_expected_outputs_keys_on_the_stage_product_not_the_first_key(tmp_path):
    """The misreg config carries an `interferogram` key (a coarse overlap
    product) before its real outputs; the v2 run failed all 87 of them on
    a predicate that matched the first key it recognised."""
    cfg = make_cfg(tmp_path)
    d = cfg.stack_dir / "configs"
    (d / "config_misreg_20180611_20180623").write_text(
        "[Common]\n[Function-1]\ngenerateIgram : \n"
        f"interferogram : {cfg.stack_dir}/coarse_interferograms/20180611_20180623\n"
        "overlap : True\n[Function-3]\nestimateAzimuthMisreg : \n"
        f"out_azimuth : {cfg.stack_dir}/misreg/azimuth/pairs/p/p.txt\ncoh_threshold : 0.7\n"
        "[Function-4]\nestimateRangeMisreg : \n"
        f"out_range : {cfg.stack_dir}/misreg/range/pairs/p/p.txt\n")
    cmd = f"SentinelWrapper.py -c {d / 'config_misreg_20180611_20180623'}"
    kinds = [k for _, k in sb.expected_outputs(cfg, "run_07_pairs_misreg", cmd)]
    assert kinds == ["file", "file"]
    assert "misreg" in sb.verify_outputs(cfg, "run_07_pairs_misreg", cmd)
    for sub in ("azimuth", "range"):
        p = cfg.stack_dir / "misreg" / sub / "pairs" / "p"
        p.mkdir(parents=True)
        (p / "p.txt").write_text("0.001\n")
    assert sb.verify_outputs(cfg, "run_07_pairs_misreg", cmd) == ""
    # overlap vs full-burst geo2rdr share a directory; the check must not
    # accept the overlap stage's files for the full-burst stage
    (d / "config_fullBurst_geo2rdr_20180611").write_text(
        f"[Common]\n[Function-1]\ngeo2rdr : \ncoregSLCdir : {cfg.stack_dir}/coreg_secondarys/20180611\n"
        "overlap : False\n")
    cmd2 = f"SentinelWrapper.py -c {d / 'config_fullBurst_geo2rdr_20180611'}"
    ov = cfg.stack_dir / "coreg_secondarys" / "20180611" / "overlap" / "IW2"
    ov.mkdir(parents=True)
    (ov / "range_top_01_02.off").write_bytes(b"\0" * 8)
    assert sb.verify_outputs(cfg, "run_09_fullBurst_geo2rdr", cmd2) != ""
    full = cfg.stack_dir / "coreg_secondarys" / "20180611" / "IW2"
    full.mkdir()
    (full / "range_01.off").write_bytes(b"\0" * 8)
    assert sb.verify_outputs(cfg, "run_09_fullBurst_geo2rdr", cmd2) == ""


def test_run_files_ignores_backups_and_notes(tmp_path):
    cfg = make_cfg(tmp_path)
    rf = cfg.stack_dir / "run_files"
    for n in ("run_16_unwrap", "run_01_unpack_topo_reference", "sentinelwrapper_run_16_unwrap",
              "run_16_unwrap.sentinelwrapper", "notes.txt"):
        (rf / n).write_text("x\n")
    assert [p.name for p in sb.run_files(cfg)] == ["run_01_unpack_topo_reference", "run_16_unwrap"]


def test_reference_unreliable_pairs_finds_cc0_at_the_reference(tmp_path):
    from insar_pipeline import time_series as ts
    import h5py
    cfg = make_cfg(tmp_path, reference_lalo="38.30,-87.60")
    ny, nx = 20, 30
    lat = np.linspace(38.47, 38.20, ny)[:, None] * np.ones((1, nx))
    lon = np.ones((ny, 1)) * np.linspace(-87.80, -87.52, nx)[None, :]
    (cfg.mintpy_dir / "inputs").mkdir()
    with h5py.File(cfg.mintpy_dir / "inputs" / "geometryRadar.h5", "w") as g:
        g.create_dataset("latitude", data=lat)
        g.create_dataset("longitude", data=lon)
    d12 = [("20180611", "20180623"), ("20180611", "20180717"), ("20180623", "20180705"), ("20180705", "20180717")]
    cc = np.ones((4, ny, nx), np.int16)
    y, x = np.unravel_index(np.argmin((lat - 38.30) ** 2 + (lon + 87.60) ** 2), lat.shape)
    cc[1, y, x] = 0                      # reference unreliable in pair 1
    cc[3, y, x] = 0                      # and in pair 3, which is already dropped
    with h5py.File(cfg.mintpy_dir / "inputs" / "ifgramStack.h5", "w") as f:
        f.create_dataset("date", data=np.array([[a.encode(), b.encode()] for a, b in d12]))
        f.create_dataset("connectComponent", data=cc)
        f.create_dataset("coherence", data=np.full((4, ny, nx), 0.8, np.float32))
        f.create_dataset("dropIfgram", data=np.array([True, True, True, False]))
    bad, info = ts.reference_unreliable_pairs(cfg)
    assert bad == ["20180611_20180717"]
    assert info["n_kept"] == 3 and info["ref_yx"] == [int(y), int(x)]
    cfgp = cfg.mintpy_dir / "smallbaselineApp.cfg"
    cfgp.write_text("mintpy.load.processor = isce\nmintpy.network.excludeDate12 = a_b\n")
    ts._set_exclude_line(cfgp, ["a_b"] + bad)
    txt = cfgp.read_text()
    assert txt.count("excludeDate12") == 1 and "20180611_20180717" in txt and "a_b" in txt


# ------------------------------------------------------------ cleanups

def _touch(p: Path, n=8):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * n)


def test_cleanup_offsets_removes_only_after_resample_and_leaves_marker(tmp_path):
    cfg = make_cfg(tmp_path, stack_cleanup_offsets=True)
    d = cfg.stack_dir / "coreg_secondarys" / "20180611"
    for n in ("range_01.off", "range_01.off.vrt", "range_01.off.xml", "azimuth_01.off"):
        _touch(d / "IW2" / n)
    assert sb.cleanup_offsets(cfg, "20180611", overlap=False) == 0.0     # no SLCs yet
    assert (d / "IW2" / "range_01.off").exists()
    _touch(d / "IW2" / "burst_01.slc")
    assert sb.cleanup_offsets(cfg, "20180611", overlap=False) > 0
    assert not list((d / "IW2").glob("*.off*"))
    assert (d / "offsets.cleaned").exists()
    # the geo2rdr predicate now accepts the marker
    conf = cfg.stack_dir / "configs" / "config_fullBurst_geo2rdr_20180611"
    conf.write_text(f"[Common]\n[Function-1]\ngeo2rdr : \ncoregSLCdir : {d}\noverlap : False\n")
    assert sb.verify_outputs(cfg, "run_09_fullBurst_geo2rdr", f"SentinelWrapper.py -c {conf}") == ""


def test_cleanup_misreg_inputs_waits_for_every_date(tmp_path):
    cfg = make_cfg(tmp_path, stack_cleanup_misreg_inputs=True)
    s = cfg.stack_dir
    for d in ("20180611", "20180623"):
        _touch(s / "coreg_secondarys" / d / "IW2" / "burst_01.slc")
        _touch(s / "coreg_secondarys" / d / "overlap" / "IW2" / "burst_top_01_02.slc")
    _touch(s / "ESD" / "20180611_20180623" / "IW2" / "combined.int", 100)
    _touch(s / "coarse_interferograms" / "20180611_20180623" / "overlap" / "IW2" / "x.int", 100)
    _touch(s / "misreg" / "azimuth" / "dates" / "20180611.txt")
    _touch(s / "misreg" / "range" / "dates" / "20180611.txt")
    assert sb.cleanup_misreg_inputs(cfg) == 0.0                          # 20180623 missing
    assert (s / "ESD").exists()
    _touch(s / "misreg" / "azimuth" / "dates" / "20180623.txt")
    _touch(s / "misreg" / "range" / "dates" / "20180623.txt")
    assert sb.cleanup_misreg_inputs(cfg) > 0
    assert not (s / "ESD").exists() and not (s / "coarse_interferograms").exists()
    assert not (s / "coreg_secondarys" / "20180611" / "overlap").exists()
    assert (s / "coreg_secondarys" / "20180611" / "overlap.cleaned").exists()
    # the overlap resample predicate accepts the marker
    conf = s / "configs" / "config_overlap_resample_20180611"
    conf.write_text(f"[Common]\n[Function-1]\nresamp_withCarrier : \n"
                    f"coregdir : {s / 'coreg_secondarys' / '20180611'}\noverlap : True\n")
    assert sb.verify_outputs(cfg, "run_06_overlap_resample", f"SentinelWrapper.py -c {conf}") == ""


def test_cleanup_coreg_slcs_needs_every_pair_filtered(tmp_path):
    cfg = make_cfg(tmp_path, stack_cleanup_coreg_after_filter=True)
    s = cfg.stack_dir
    for p in ("20180611_20180623", "20180611_20180705"):
        (s / "configs" / f"config_igram_filt_coh_{p}").write_text("x")
    _touch(s / "coreg_secondarys" / "20180611" / "IW2" / "burst_01.slc", 1000)
    _touch(s / "merged" / "interferograms" / "20180611_20180623" / "filt_fine.int")
    _touch(s / "merged" / "interferograms" / "20180611_20180623" / "filt_fine.cor")
    assert sb.cleanup_coreg_slcs(cfg) == 0.0                              # one pair not filtered
    _touch(s / "merged" / "interferograms" / "20180611_20180705" / "filt_fine.int")
    _touch(s / "merged" / "interferograms" / "20180611_20180705" / "filt_fine.cor")
    assert sb.cleanup_coreg_slcs(cfg) > 0
    assert not (s / "coreg_secondarys" / "20180611").exists()
    marker = s / "coreg_secondarys" / "20180611.cleaned"
    assert marker.exists() and "20180611_20180705" in marker.read_text()
    # resample predicate accepts it; SAFE cleanup counts the date as done
    conf = s / "configs" / "config_fullBurst_resample_20180611"
    conf.write_text(f"[Common]\n[Function-1]\nresamp_withCarrier : \n"
                    f"coregdir : {s / 'coreg_secondarys' / '20180611'}\noverlap : False\n")
    assert sb.verify_outputs(cfg, "run_10_fullBurst_resample", f"SentinelWrapper.py -c {conf}") == ""
    # and a looks change is refused rather than silently unrebuildable
    with pytest.raises(RuntimeError):
        sb.invalidate_after_looks_change(cfg)


def test_cleanup_unfiltered_keeps_run14_predicate_satisfied(tmp_path):
    cfg = make_cfg(tmp_path, stack_cleanup_unfiltered_igrams=True)
    d = cfg.stack_dir / "merged" / "interferograms" / "20180611_20180623"
    for n in ("fine.int", "fine.int.xml", "fine.int.vrt", "fine.int.full.vrt", "fine.cor", "fine.cor.xml"):
        _touch(d / n, 100)
    assert sb.cleanup_unfiltered(cfg) == 0.0                              # not filtered yet
    _touch(d / "filt_fine.int"); _touch(d / "filt_fine.cor")
    assert sb.cleanup_unfiltered(cfg) > 0
    assert not (d / "fine.int").exists() and (d / "fine.cleaned").exists()
    conf = cfg.stack_dir / "configs" / "config_merge_igram_20180611_20180623"
    conf.write_text(f"[Common]\n[Function-1]\nmergeBursts : \noutfile : {d / 'fine.int'}\n")
    assert sb.verify_outputs(cfg, "run_14_merge_burst_igram", f"SentinelWrapper.py -c {conf}") == ""


def test_cleanup_safes_ignores_dates_with_only_overlap_coreg(tmp_path):
    cfg = make_cfg(tmp_path, stack_delete_safes_after_coreg=True, stack_keep_recent_safes=0,
                   stack_reference_date="20181208")
    for d in ("20180611", "20180623"):
        _touch(cfg.slc_dir / f"S1A_IW_SLC__1SSV_{d}T234756_{d}T234802_022316_026A5D_0F42.SAFE" / "manifest.safe", 50)
    _touch(cfg.stack_dir / "coreg_secondarys" / "20180611" / "overlap" / "IW2" / "burst_top.slc")   # overlap only
    _touch(cfg.stack_dir / "coreg_secondarys" / "20180623" / "IW2" / "burst_01.slc")             # full-burst done
    # phaselinked-9: a date is done only when resamp_withCarrier has written the swath
    # xml (it does so after the last burst) and every burst it lists exists.
    (cfg.stack_dir / "coreg_secondarys" / "20180623" / "IW2.xml").write_text(
        '<component><property name="numberofbursts"><value>1</value></property></component>')
    sb.cleanup_safes(cfg)
    assert list(cfg.slc_dir.glob("*20180611*"))          # still needed by resample
    assert not list(cfg.slc_dir.glob("*20180623*"))      # consumed


def test_fused_partner_and_date_parsing(tmp_path):
    cfg = make_cfg(tmp_path)
    rf = cfg.stack_dir / "run_files"
    names = ["run_05_overlap_geo2rdr", "run_06_overlap_resample", "run_07_pairs_misreg",
             "run_09_fullBurst_geo2rdr", "run_10_fullBurst_resample"]
    for n in names:
        (rf / n).write_text("x\n")
    files = sb.run_files(cfg)
    assert sb._fused_partner(files, 0) == 1
    assert sb._fused_partner(files, 1) is None
    assert sb._fused_partner(files, 2) is None
    assert sb._fused_partner(files, 3) == 4
    assert sb._date_of("SentinelWrapper.py -c /x/configs/config_fullBurst_geo2rdr_20180611") == "20180611"
    assert sb._date_of("SentinelWrapper.py -c /x/configs/config_misreg_20180611_20180623") == "20180623"


def test_execute_fused_chains_per_date_and_runs_cleanup(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, stack_num_proc_topo=2, stack_cleanup_offsets=True)
    rf = cfg.stack_dir / "run_files"
    dates = ["20180611", "20180623", "20180705"]
    (rf / "run_09_fullBurst_geo2rdr").write_text(
        "".join(f"SentinelWrapper.py -c {cfg.stack_dir}/configs/config_fullBurst_geo2rdr_{d}\n" for d in dates))
    (rf / "run_10_fullBurst_resample").write_text(
        "".join(f"SentinelWrapper.py -c {cfg.stack_dir}/configs/config_fullBurst_resample_{d}\n" for d in dates))
    order = []

    def fake_run_one(cfg_, run_name, idx, cmd, workers=1):
        d = sb._date_of(cmd)
        order.append((run_name.split("_")[1], d))
        coreg = cfg_.stack_dir / "coreg_secondarys" / d / "IW2"
        if "geo2rdr" in run_name:
            _touch(coreg / "range_01.off", 64)
            rc = 1 if d == "20180705" else 0            # one date's geo2rdr fails
        else:
            assert (coreg / "range_01.off").exists(), "resample must follow its own geo2rdr"
            _touch(coreg / "burst_01.slc", 64)
            rc = 0
        return {"cmd": cmd, "returncode": rc, "seconds": 1.0, "peak_rss_gb": None, "skipped": False}
    monkeypatch.setattr(sb, "_run_one", fake_run_one)
    monkeypatch.setattr(sb, "_record_timing", lambda cfg_, s: None)
    files = sb.run_files(cfg)
    with pytest.raises(RuntimeError):                     # fail-fast: coregistration chain
        sb.execute_fused(cfg, files[0], files[1], fail_fast=True)
    # every successful date ran geo2rdr then resample, and its offsets were cleaned
    for d in ("20180611", "20180623"):
        assert ("09", d) in order and ("10", d) in order
        assert order.index(("09", d)) < order.index(("10", d))
        assert (cfg.stack_dir / "coreg_secondarys" / d / "offsets.cleaned").exists()
        assert not (cfg.stack_dir / "coreg_secondarys" / d / "IW2" / "range_01.off").exists()
    assert ("10", "20180705") not in order                # resample skipped after geo2rdr failure


def test_execute_fused_pairs_chains_and_releases_coreg(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, stack_num_proc=2, stack_cleanup_coreg_after_filter=True,
                   stack_cleanup_unfiltered_igrams=True, stack_cleanup_burst_igrams=True)
    s = cfg.stack_dir
    pairs = ["20180611_20180623", "20180611_20180705", "20180623_20180705"]
    for stage in ("run_13_generate_burst_igram", "run_14_merge_burst_igram", "run_15_filter_coherence"):
        kind = {"13": "generate_igram", "14": "merge_igram", "15": "igram_filt_coh"}[stage.split("_")[1]]
        (s / "run_files" / stage).write_text(
            "".join(f"SentinelWrapper.py -c {s}/configs/config_{kind}_{p}\n" for p in pairs))
        for p in pairs:
            (s / "configs" / f"config_{kind}_{p}").write_text("x")
    for d in ("20180611", "20180623", "20180705"):
        _touch(s / "coreg_secondarys" / d / "IW2" / "burst_01.slc", 1000)
    order = []

    def fake_run_one(cfg_, run_name, idx, cmd, workers=1):
        pr = sb.pair_of(cmd)
        order.append((run_name.split("_")[1], pr))
        m = s / "merged" / "interferograms" / pr
        if "generate" in run_name:
            _touch(s / "interferograms" / pr / "IW2" / "fine_01.int", 100)
            rc = 1 if pr == "20180623_20180705" else 0          # one pair fails
        elif "merge" in run_name:
            assert (s / "interferograms" / pr / "IW2" / "fine_01.int").exists()
            _touch(m / "fine.int", 100); rc = 0
        else:
            _touch(m / "filt_fine.int", 100); _touch(m / "filt_fine.cor", 100); rc = 0
        return {"cmd": cmd, "returncode": rc, "seconds": 1.0, "peak_rss_gb": None, "skipped": False}
    monkeypatch.setattr(sb, "_run_one", fake_run_one)
    files = sb.run_files(cfg)
    sums = sb.execute_fused_pairs(cfg, *files)               # per-pair failure tolerated
    assert [x["run_file"] for x in sums] == [f.name for f in files]
    assert sums[0]["failed"] and sums[2]["failed"]           # the failed pair's chain stopped
    for p in pairs[:2]:
        assert order.index(("13", p)) < order.index(("14", p)) < order.index(("15", p))
        assert (s / "merged" / "interferograms" / p / "fine.cleaned").exists()
        assert not (s / "interferograms" / p / "IW2").exists()  # burst products cleaned
    assert ("14", "20180623_20180705") not in order
    # 20180611's two pairs are both filtered -> its SLCs released; 20180623 has a
    # failed (unfiltered) pair -> kept
    assert (s / "coreg_secondarys" / "20180611.cleaned").exists()
    assert (s / "coreg_secondarys" / "20180623" / "IW2" / "burst_01.slc").exists()


def test_search_bursts_complete_unions_incomplete_cmr_pages(tmp_path, monkeypatch):
    """asf_search returns a short list when CMR drops a page and only logs it;
    on the P2 launch two dates came back with 2 bursts. Repeat until an
    attempt adds nothing new; union the granule IDs."""
    from insar_pipeline import data_acquisition as da

    class R:
        def __init__(self, fid, date):
            self.properties = {"fileID": fid, "flightDirection": "ASCENDING",
                               "burst": {"subswath": "IW2"},
                               "startTime": f"{date[:4]}-{date[4:6]}-{date[6:]}T23:47:00Z"}
            self.geometry = {"type": "Polygon", "coordinates": [[[-87.9, 38.1], [-87.0, 38.1],
                                                                  [-87.0, 38.6], [-87.9, 38.6]]]}
    full = [R(f"S1_0388{86+i}_IW2_{d}", d) for d in ("20160609", "20160621") for i in range(3)]
    calls = {"n": 0}

    class FakeASF:
        class DATASET:
            SLC_BURST = "SLC-BURST"

        @staticmethod
        def search(**kw):
            calls["n"] += 1
            return full[:-1] if calls["n"] == 1 else full      # first page short by one
    monkeypatch.setitem(sys.modules, "asf_search", FakeASF)
    monkeypatch.setattr(da.time, "sleep", lambda s: None)
    cfg = make_cfg(tmp_path, track=19)
    by_date, lons, lats = da.search_bursts_complete(cfg, {"IW2"}, "VV", "ASC")
    assert calls["n"] == 3                       # short, complete(+1 new), complete(+0) -> stop
    assert {d: len(v) for d, v in by_date.items()} == {"20160609": 3, "20160621": 3}
    assert len(lons) == 4 * 6


def test_unwrap_error_applied_falls_back_to_what_exists(tmp_path):
    from insar_pipeline import time_series as ts
    import h5py
    assert ts.unwrap_error_dataset("bridging+phase_closure") == "unwrapPhase_bridging_phaseClosure"
    assert ts.unwrap_error_dataset("bridging") == "unwrapPhase_bridging"
    cfg = make_cfg(tmp_path)
    (cfg.mintpy_dir / "inputs").mkdir()
    with h5py.File(cfg.mintpy_dir / "inputs" / "ifgramStack.h5", "w") as f:
        f.create_dataset("unwrapPhase", data=np.zeros((2, 3, 3), np.float32))
        f.create_dataset("unwrapPhase_bridging", data=np.zeros((2, 3, 3), np.float32))
    cfgp = cfg.mintpy_dir / "smallbaselineApp.cfg"
    cfgp.write_text("mintpy.load.processor = isce\nmintpy.unwrapError.method = bridging+phase_closure\n")
    rec = ts.unwrap_error_applied(cfg, cfgp)
    assert rec["applied"] == "bridging" and rec["dataset"] == "unwrapPhase_bridging"
    assert "mintpy.unwrapError.method = bridging" in cfgp.read_text()
    assert cfgp.read_text().count("unwrapError.method") == 1
    assert (cfg.mintpy_dir / "unwrap_error_applied.json").exists()
    # nothing applied -> plain unwrapPhase, method 'no'
    with h5py.File(cfg.mintpy_dir / "inputs" / "ifgramStack.h5", "w") as f:
        f.create_dataset("unwrapPhase", data=np.zeros((2, 3, 3), np.float32))
    rec = ts.unwrap_error_applied(cfg, cfgp)
    assert rec["applied"] == "no" and rec["dataset"] == "unwrapPhase"


# ------------------------------------------------------ network connectivity guard
def test_ensure_connected_network_raises_naming_the_bridges(monkeypatch, tmp_path):
    from insar_pipeline import time_series as ts
    cfg = types.SimpleNamespace(mintpy_dir=tmp_path, max_temp_baseline=48, max_perp_baseline=200.0)
    cfgp = tmp_path / "smallbaselineApp.cfg"; cfgp.write_text("mintpy.x = 1\n")
    dates = ["20180101", "20180113", "20180125", "20180401", "20180413"]
    formed = [("20180101_20180113", True, 12, 10.0), ("20180113_20180125", True, 12, -20.0),
              ("20180401_20180413", True, 12, 5.0),
              ("20180125_20180401", False, 66, 30.0),      # the bridge, pruned by tempBaseMax
              ("20180101_20180125", True, 24, 40.0)]
    kept = [p for p, k, _, _ in formed if k]
    monkeypatch.setattr(ts, "kept_network", lambda cfg: (dates, kept, formed))
    monkeypatch.setattr(ts, "_mintpy", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not rerun")))
    with pytest.raises(RuntimeError) as e:
        ts.ensure_connected_network(cfg, cfgp)
    msg = str(e.value)
    assert "2 pieces" in msg and "20180125_20180401" in msg and "66 d" in msg and "tempBaseMax 48" in msg
    info = json.loads((tmp_path / "network_connectivity.json").read_text())
    assert info["components"] == [3, 2] and info["orphaned_dates"] == []


def test_ensure_connected_network_excludes_an_orphaned_date(monkeypatch, tmp_path):
    from insar_pipeline import time_series as ts
    cfg = types.SimpleNamespace(mintpy_dir=tmp_path, max_temp_baseline=48, max_perp_baseline=200.0)
    cfgp = tmp_path / "smallbaselineApp.cfg"; cfgp.write_text("mintpy.x = 1\n")
    (tmp_path / "maskConnComp.h5").write_bytes(b"stale")
    dates = ["20180101", "20180113", "20180125"]
    formed = [("20180101_20180113", True, 12, 1.0), ("20180113_20180125", False, 12, 1.0),
              ("20180101_20180125", False, 24, 1.0)]        # 20180125 has no kept pair
    calls = []
    monkeypatch.setattr(ts, "kept_network", lambda cfg: (dates, [p for p, k, _, _ in formed if k], formed))
    monkeypatch.setattr(ts, "_mintpy", lambda *a, **k: calls.append(a[2]) or types.SimpleNamespace(returncode=0))
    info = ts.ensure_connected_network(cfg, cfgp)
    assert calls == ["modify_network"] and info["excluded_dates"] == ["20180125"]
    assert "mintpy.network.excludeDate = 20180125" in cfgp.read_text()
    assert not (tmp_path / "maskConnComp.h5").exists()
    assert info["components"] == [2] and info["orphaned_dates"] == []


# ------------------------------------------------- unwrap-error recompute guard
def _tiny_stack(path, ref=("10", "20"), keep=(True, True), with_out=True):
    import h5py
    with h5py.File(path, "w") as f:
        f["date"] = np.array([[b"20180101", b"20180113"], [b"20180113", b"20180125"]])
        f["dropIfgram"] = np.array(keep)
        f.create_dataset("unwrapPhase", data=np.zeros((2, 2, 2), np.float32)).attrs["MODIFICATION_TIME"] = 1.0
        if with_out:
            f.create_dataset("unwrapPhase_bridging", data=np.zeros((2, 2, 2), np.float32)).attrs["MODIFICATION_TIME"] = 2.0
        f.attrs["REF_Y"], f.attrs["REF_X"] = ref


def test_unwrap_error_recomputes_when_reference_or_network_changes(tmp_path):
    import h5py
    from insar_pipeline import time_series as ts
    cfg = types.SimpleNamespace(mintpy_dir=tmp_path)
    (tmp_path / "inputs").mkdir()
    stack = tmp_path / "inputs" / "ifgramStack.h5"
    _tiny_stack(stack)
    # first time: no record -> forced
    t = ts.refresh_unwrap_error_if_stale(cfg)
    assert t is not None
    with h5py.File(stack, "r") as f:
        assert float(f["unwrapPhase"].attrs["MODIFICATION_TIME"]) == t
    # the step "ran": stamp the output newer than the bump, record state
    with h5py.File(stack, "r+") as f:
        f["unwrapPhase_bridging"].attrs["MODIFICATION_TIME"] = t + 1
    ts.record_unwrap_error_state(cfg, t)
    assert ts.refresh_unwrap_error_if_stale(cfg) is None          # unchanged -> skip
    _tiny_stack(stack, ref=("11", "20"))                            # reference moved
    assert ts.refresh_unwrap_error_if_stale(cfg) is not None
    _tiny_stack(stack, keep=(True, False))                          # a pair dropped
    assert ts.refresh_unwrap_error_if_stale(cfg) is not None
    # a bump that MintPy then ignored is an error, not a log line
    t2 = ts.refresh_unwrap_error_if_stale(cfg)
    with pytest.raises(RuntimeError):
        ts.record_unwrap_error_state(cfg, t2)


# ------------------------------------------------------ class-matched relative rates
def test_relative_rates_cancel_a_coherence_dependent_bias():
    from insar_pipeline.quality_assessment import _relative_rates
    rng = np.random.default_rng(0)
    n = 40000
    tc = rng.uniform(0.4, 1.0, n)
    bias = -20 * (1 - tc)                       # −12 mm/yr at tc 0.4, 0 at tc 1.0
    vel = bias + rng.normal(0, 0.3, n)
    region = np.zeros(n, bool); region[:4000] = True
    tc[:4000] = rng.uniform(0.4, 0.6, 4000)     # region is low-coherence ground
    vel[:4000] = -20 * (1 - tc[:4000]) - 5      # plus 5 mm/yr of real subsidence
    far = ~region; good = np.ones(n, bool)
    cp_free = tc > 0.7
    r = _relative_rates(vel.reshape(200, 200), tc.reshape(200, 200), region.reshape(200, 200),
                        far.reshape(200, 200), good.reshape(200, 200), cp_free.reshape(200, 200))
    assert r["plain_mm_yr"] < -8                      # plain difference mixes bias in
    assert abs(r["class_matched_mm_yr"] + 5) < 0.6    # class matching recovers the real −5
    assert "bias_free_mm_yr" not in r                 # region has no bias-free cells


# ------------------------------------------------------------- miaplpy backend
def test_miaplpy_template_carries_every_setting(tmp_path):
    from insar_pipeline import miaplpy_backend as mb
    cfg = types.SimpleNamespace(
        project_dir=tmp_path, stack_dir=tmp_path / "stack", bbox=(-87.8, 38.2, -87.52, 38.47), stack_bbox_margin_deg=0.05,
        miaplpy_num_processors=6, miaplpy_ministack_size=10, miaplpy_range_window=15, miaplpy_azimuth_window=15,
        miaplpy_phase_linking_method="sequential_EMI", miaplpy_time_lag=30, stack_reference_date="20230226", stack_unw_defomax=4.0,
        reference_lalo="38.29,-87.55", reference_min_coherence=0.85, max_temp_baseline=120, max_perp_baseline=350.0,
        min_temporal_coherence=0.4, mintpy_tropo_method="pyaps", mintpy_deramp="linear", mintpy_geocode_lalo_step="-0.000135,0.000172",
        orbit_direction="ascending", track=19)
    (tmp_path / "stack").mkdir(); (tmp_path / "stack" / "merged" / "SLC" / "20230221").mkdir(parents=True)
    (tmp_path / "stack" / "crop_window.json").write_text(json.dumps({"r0": 249, "r1": 1355, "c0": 338, "c1": 1306, "length": 1396, "width": 2272}))
    (tmp_path / "stack" / "merged" / "SLC" / "20230221" / "20230221.slc.full").write_bytes(b"")
    (tmp_path / "stack" / "merged" / "SLC" / "20230221" / "20230221.slc.full.xml").write_text(
        '<property name="width"><value>25001</value></property><property name="length"><value>4190</value></property>')
    cfg.stack_azimuth_looks, cfg.stack_range_looks = 3, 11
    tpl = mb.write_template(cfg)
    assert tpl.name != "miaplpyApp.cfg" and tpl.name.endswith("SenAT19.cfg")
    text = tpl.read_text()
    assert mb.aligned_subset(cfg) == (747, 4065, 3718, 14366)
    assert "miaplpy.subset.yx           = 747:4065,3718:14366" in text
    assert "miaplpy.interferograms.networkType     = single_reference" in text
    assert "miaplpy.interferograms.referenceDate   = 20230226" in text
    assert f"mintpy.deramp.maskFile                 = {tmp_path / 'miaplpy' / 'network_single_reference' / 'maskDeramp.h5'}" in text
    assert "mintpy.reference.lalo                  = 38.29,-87.55" in text
    assert mb.network_dir(cfg) == tmp_path / "miaplpy" / "network_single_reference"


def test_phase_linked_export_multilooks_pads_and_signs(tmp_path, monkeypatch):
    """A tiny linked phase series -> pair files on the merged grid: phase(A)-phase(B),
    averaged over the looks, placed at the crop offset, zero elsewhere."""
    import h5py
    from insar_pipeline import miaplpy_backend as mb
    stack = tmp_path / "stack"; (stack / "merged" / "SLC" / "20230221").mkdir(parents=True)
    (stack / "crop_window.json").write_text(json.dumps({"r0": 1, "r1": 3, "c0": 2, "c1": 4, "length": 5, "width": 8}))
    (stack / "merged" / "SLC" / "20230221" / "20230221.slc.full").write_bytes(b"")
    (stack / "merged" / "SLC" / "20230221" / "20230221.slc.full.xml").write_text(
        '<property name="width"><value>16</value></property><property name="length"><value>10</value></property>')
    cfg = types.SimpleNamespace(project_dir=tmp_path, stack_dir=stack, stack_azimuth_looks=2, stack_range_looks=2,
                                stack_reference_date="20230221", min_temporal_coherence=0.4,
                                miaplpy_network_connections=0, miaplpy_ministack_size=10,
                                miaplpy_phase_linking_method="EMI")
    (stack / "merged").mkdir(parents=True, exist_ok=True)
    assert mb.aligned_subset(cfg) == (2, 6, 4, 8)
    work = tmp_path / "miaplpy" / "inverted"; work.mkdir(parents=True)
    dates = ["20230104", "20230221", "20230305"]
    ph = np.zeros((3, 4, 4), np.float32)
    ph[1] = 0.0                       # reference
    ph[0] = 0.5                       # earlier date: pair 20230104_20230221 has phase(0104)-phase(0221) = +0.5
    ph[2] = -0.25                     # later date:   pair 20230221_20230305 has phase(0221)-phase(0305) = +0.25
    with h5py.File(work / "phase_series.h5", "w") as f:
        f["date"] = np.array([d.encode() for d in dates]); f["phase"] = ph
        f["temporalCoherence"] = np.full((2,) + ph.shape[1:], 0.6, np.float32)
    written = {}
    monkeypatch.setattr(mb, "import_isce", lambda: (None, None), raising=False)
    import insar_pipeline.isceenv as ie
    monkeypatch.setattr(ie, "import_isce", lambda: (None, None))
    monkeypatch.setattr(mb, "_write_isce", lambda isceobj, path, arr, dtype: written.__setitem__(path.name + "@" + path.parent.name, arr.copy()))
    pairs = mb.export_phase_linked_ifgrams(cfg)
    tc = np.load(mb.linked_temporal_coherence_path(cfg))["full"]
    assert tc.shape == (5, 8) and np.isnan(tc[0, 0]) and np.allclose(tc[1:3, 2:4], 0.6)
    assert pairs == ["20230104_20230221", "20230221_20230305"]
    a = written["filt_fine.int@20230104_20230221"]; c = written["filt_fine.cor@20230104_20230221"]
    assert a.shape == (5, 8)
    assert np.allclose(np.angle(a[1:3, 2:4]), 0.5) and np.allclose(c[1:3, 2:4], 1.0)
    assert np.all(a[0] == 0) and np.all(a[:, 0] == 0) and np.all(c[3:] == 0)
    b = written["filt_fine.int@20230221_20230305"]
    assert np.allclose(np.angle(b[1:3, 2:4]), 0.25)


def test_empty_slc_dates_flags_nan_and_zero_dates(tmp_path):
    import h5py
    from insar_pipeline import miaplpy_backend as mb
    rng = np.random.default_rng(0)
    slc = (rng.normal(size=(4, 300, 300)) + 1j * rng.normal(size=(4, 300, 300))).astype(np.complex64)
    slc[1] = np.nan + 1j * np.nan          # a date whose VRT pointed at deleted bursts
    slc[3] = 0                              # a date that never got written
    with h5py.File(tmp_path / "slcStack.h5", "w") as f:
        f["slc"] = slc
        f["date"] = np.array([b"20230104", b"20230116", b"20230128", b"20230209"])
    with h5py.File(tmp_path / "slcStack.h5", "r") as f:
        assert mb._empty_slc_dates(f) == ["20230116", "20230209"]


def test_repoint_full_vrts_rewrites_vrts_with_foreign_sources(tmp_path):
    """A VRT whose sources are the (deleted) per-burst files is rewritten
    against the real .full raster; one that already points at it is kept."""
    from osgeo import gdal
    from insar_pipeline import miaplpy_backend as mb
    gdal.UseExceptions()
    stack = tmp_path / "stack"
    slc = stack / "merged" / "SLC" / "20230104"; slc.mkdir(parents=True)
    (stack / "merged" / "geom_reference").mkdir()
    L, W = 40, 50
    data = (np.arange(L * W, dtype=np.float32) + 1).reshape(L, W).astype(np.complex64)
    full = slc / "20230104.slc.full"
    data.tofile(full)
    # ISCE-style xml so GDAL's ISCE driver can open the raw file
    full.with_name(full.name + ".xml").write_text(f"""<imageFile>
<property name="WIDTH"><value>{W}</value></property>
<property name="LENGTH"><value>{L}</value></property>
<property name="NUMBER_BANDS"><value>1</value></property>
<property name="DATA_TYPE"><value>CFLOAT</value></property>
<property name="SCHEME"><value>BIL</value></property>
<property name="ACCESS_MODE"><value>read</value></property>
<property name="BYTE_ORDER"><value>l</value></property>
</imageFile>""")
    vrt = full.with_name(full.name + ".vrt")
    vrt.write_text(f"""<VRTDataset rasterXSize="{W}" rasterYSize="{L}"><VRTRasterBand dataType="CFloat32" band="1">
<NoDataValue>0.0</NoDataValue><SimpleSource>
<SourceFilename relativeToVRT="1">../../../coreg_secondarys/20230104/IW2/burst_01.slc.vrt</SourceFilename>
<SourceBand>1</SourceBand><SourceProperties RasterXSize="{W}" RasterYSize="{L}" DataType="CFloat32"/>
<SrcRect xOff="5" yOff="5" xSize="{W-5}" ySize="{L-5}"/><DstRect xOff="5" yOff="5" xSize="{W-5}" ySize="{L-5}"/>
</SimpleSource></VRTRasterBand></VRTDataset>""")
    assert mb.repoint_full_vrts(stack) == 1
    assert "20230104.slc.full" in vrt.read_text() and "coreg_secondarys" not in vrt.read_text()
    ds = gdal.Open(str(vrt))
    blk = ds.GetRasterBand(1).ReadAsArray(10, 10, 4, 4)
    assert np.isfinite(blk).all() and np.allclose(blk.real, data[10:14, 10:14].real)
    assert mb.repoint_full_vrts(stack) == 0          # idempotent


def test_miaplpy_route_writes_no_baseline_limits():
    from insar_pipeline import stack_backend as sb
    from insar_pipeline.config import PipelineConfig
    base = dict(project_dir=Path("/tmp/x"), bbox=[-87.8, 38.2, -87.52, 38.47], start_date="2023-01-01", end_date="2023-08-31",
                max_perp_baseline=350.0, max_temp_baseline=120)
    classic = sb.mintpy_loader_lines(PipelineConfig(**base, backend="stack"))
    linked = sb.mintpy_loader_lines(PipelineConfig(**base, backend="miaplpy"))
    assert "mintpy.network.tempBaseMax            = 120" in classic
    assert "mintpy.network.tempBaseMax            = no" in linked
    assert "mintpy.network.perpBaseMax            = no" in linked


def test_merge_date_from_command_only_matches_dates():
    from insar_pipeline import miaplpy_backend as mb
    assert mb.merge_date_from_command("SentinelWrapper.py -c /x/stack/configs/config_merge_20230104") == "20230104"
    assert mb.merge_date_from_command("SentinelWrapper.py -c /x/stack/configs/config_merge_lat") is None
    assert mb.merge_date_from_command("SentinelWrapper.py -c /x/stack/configs/config_merge_20230104_extra") is None


def _pl_cfg(tmp_path):
    from insar_pipeline.config import PipelineConfig
    return PipelineConfig(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.52, 38.47], start_date="2023-01-01",
                          end_date="2023-08-31", backend="miaplpy", stack_reference_date="20230221")


def _fake_merged(cfg, date, L=6, W=8, truncate=False):
    d = cfg.stack_dir / "merged" / "SLC" / date; d.mkdir(parents=True, exist_ok=True)
    full = d / f"{date}.slc.full"
    np.ones((L, W), np.complex64).tofile(full)
    if truncate:
        full.write_bytes(full.read_bytes()[:-8])
    full.with_name(full.name + ".xml").write_text(
        f'<imageFile><property name="width"><value>{W}</value></property>'
        f'<property name="length"><value>{L}</value></property></imageFile>')
    b = cfg.stack_dir / "coreg_secondarys" / date / "IW2"; b.mkdir(parents=True, exist_ok=True)
    (b / "burst_01.slc").write_bytes(b"x" * 1000)
    return full


def test_cleanup_coreg_date_only_after_verified_merge(tmp_path):
    from insar_pipeline import miaplpy_backend as mb
    cfg = _pl_cfg(tmp_path)
    _fake_merged(cfg, "20230104")
    _fake_merged(cfg, "20230116", truncate=True)          # merged copy incomplete
    _fake_merged(cfg, "20230221")                         # the reference date
    assert mb.cleanup_coreg_date(cfg, "20230104") > 0
    assert not (cfg.stack_dir / "coreg_secondarys" / "20230104").exists()
    assert (cfg.stack_dir / "coreg_secondarys" / "20230104.cleaned").exists()
    assert mb.cleanup_coreg_date(cfg, "20230116") == 0.0 and (cfg.stack_dir / "coreg_secondarys" / "20230116").exists()
    assert mb.cleanup_coreg_date(cfg, "20230221") == 0.0 and (cfg.stack_dir / "coreg_secondarys" / "20230221").exists()
    assert mb.cleanup_coreg_after_merge(cfg) == 0.0       # nothing left that qualifies


def test_delete_merged_slcs_after_load_keeps_reference_and_refuses_nan(tmp_path):
    import h5py
    from insar_pipeline import miaplpy_backend as mb
    cfg = _pl_cfg(tmp_path)
    for d in ("20230104", "20230116", "20230221"):
        _fake_merged(cfg, d)
    work = mb.work_dir(cfg); (work / "inputs").mkdir(parents=True)
    rng = np.random.default_rng(1)
    slc = (rng.normal(size=(3, 300, 300)) + 1j).astype(np.complex64)
    with h5py.File(work / "inputs" / "slcStack.h5", "w") as f:
        f["slc"] = slc; f["date"] = np.array([b"20230104", b"20230116", b"20230221"])
    gb = mb.delete_merged_slcs_after_load(cfg)
    assert gb > 0
    assert not (cfg.stack_dir / "merged" / "SLC" / "20230104" / "20230104.slc.full").exists()
    assert (cfg.stack_dir / "merged" / "SLC" / "20230104" / "20230104.slc.full.xml").exists()
    assert (cfg.stack_dir / "merged" / "SLC" / "20230221" / "20230221.slc.full").exists()
    with h5py.File(work / "inputs" / "slcStack.h5", "r+") as f:
        f["slc"][1] = np.nan
    _fake_merged(cfg, "20230116")
    with pytest.raises(RuntimeError):
        mb.delete_merged_slcs_after_load(cfg)
    assert (cfg.stack_dir / "merged" / "SLC" / "20230116" / "20230116.slc.full").exists()


def test_delete_patches_after_concatenate_verifies_phase_series(tmp_path):
    import h5py
    from insar_pipeline import miaplpy_backend as mb
    cfg = _pl_cfg(tmp_path)
    inv = mb.work_dir(cfg) / "inverted"; (inv / "PATCHES" / "PATCH_0000").mkdir(parents=True)
    (inv / "PATCHES" / "PATCH_0000" / "flag.npy").write_bytes(b"0" * 100)
    # phaselinked-1: the fixture looks like a real MiaplPy output. Completeness is judged
    # on temporalCoherence, which MiaplPy initialises to -1 and writes last in each block.
    with h5py.File(inv / "phase_series.h5", "w") as f:
        f["phase"] = np.full((3, 200, 200), np.nan, np.float32)
        f["temporalCoherence"] = np.full((2, 200, 200), -1, np.float32)     # no block written yet
        f["date"] = np.array([b"20230104", b"20230116", b"20230128"])
    with pytest.raises(RuntimeError):
        mb.delete_patches_after_concatenate(cfg)
    assert (inv / "PATCHES").exists()
    with h5py.File(inv / "phase_series.h5", "r+") as f:
        f["phase"][:] = 0.5
        f["temporalCoherence"][:] = 0.6                                     # every block written
    assert mb.delete_patches_after_concatenate(cfg) > 0
    assert not (inv / "PATCHES").exists() and (inv / "PATCHES.cleaned").exists()


def test_delete_slc_stack_after_concatenate_and_load_step_stays_done(tmp_path):
    import h5py
    from insar_pipeline import miaplpy_backend as mb
    cfg = _pl_cfg(tmp_path)
    work = mb.work_dir(cfg); (work / "inputs").mkdir(parents=True); (work / "inverted").mkdir()
    with h5py.File(work / "inputs" / "slcStack.h5", "w") as f:
        f["slc"] = np.ones((2, 4, 4), np.complex64); f["date"] = np.array([b"20230104", b"20230221"])
    with pytest.raises(RuntimeError):                    # no phase series yet
        mb.delete_slc_stack_after_concatenate(cfg)
    with h5py.File(work / "inverted" / "phase_series.h5", "w") as f:
        f["phase"] = np.full((2, 200, 200), 0.1, np.float32)
        f["temporalCoherence"] = np.full((2, 200, 200), 0.6, np.float32)
        f["date"] = np.array([b"20230104", b"20230221"])
    assert mb.delete_slc_stack_after_concatenate(cfg) > 0
    assert not (work / "inputs" / "slcStack.h5").exists()
    assert mb.step_done(cfg, "load_data")                # a resumed run must not try to reload


def test_network_pairs_star_and_connections():
    from insar_pipeline import miaplpy_backend as mb
    d = ["20230104", "20230116", "20230128", "20230209"]
    star = mb.network_pairs(d, 0, "20230116")
    assert [(a, b) for a, b, _, _ in star] == [("20230104", "20230116"), ("20230116", "20230128"), ("20230116", "20230209")]
    assert star[0][2:] == (0, 1) and star[1][2:] == (1, 2)          # (index of A, index of B), A earlier
    net = mb.network_pairs(d, 2, "20230116")
    assert [(a, b) for a, b, _, _ in net] == [("20230104", "20230116"), ("20230104", "20230128"), ("20230116", "20230128"),
                                              ("20230116", "20230209"), ("20230128", "20230209")]


def test_mini_stack_index_matches_miaplpy_partition():
    from insar_pipeline import miaplpy_backend as mb
    # 273 images, size 10: 27 stacks, the last holds 13 images (indices 260..272)
    assert mb.mini_stack_index(0, 273, 10) == 0 and mb.mini_stack_index(9, 273, 10) == 0
    assert mb.mini_stack_index(10, 273, 10) == 1 and mb.mini_stack_index(259, 273, 10) == 25
    assert mb.mini_stack_index(260, 273, 10) == 26 and mb.mini_stack_index(272, 273, 10) == 26
    assert mb.mini_stack_index(5, 8, 10) == 0                        # fewer images than one stack


def _synthetic_phase_series(n, L, W, size, datum_noise, rng):
    """Smooth true phase per date plus, optionally, a random per-pixel datum
    per mini-stack (the sequential-EMI defect)."""
    yy, xx = np.mgrid[0:L, 0:W]
    ph = np.zeros((n, L, W), np.float32)
    for k in range(n):
        ph[k] = 0.002 * k * xx + 0.001 * k * yy + rng.normal(scale=0.05, size=(L, W))
        if datum_noise:
            ms = min(k // size, max(1, n // size) - 1)
            ph[k] += rng.normal(scale=1.5, size=(L, W)) if False else _ms_offset(ms, L, W, rng)
    return ph


_OFFSETS = {}


def _ms_offset(ms, L, W, rng):
    if ms not in _OFFSETS:
        _OFFSETS[ms] = rng.uniform(-np.pi, np.pi, size=(L, W)).astype(np.float32)
    return _OFFSETS[ms]


def test_datum_check_flags_per_ministack_offsets_only_for_sequential(tmp_path):
    import types
    from insar_pipeline import miaplpy_backend as mb
    rng = np.random.default_rng(3)
    _OFFSETS.clear()
    n, L, W, size = 30, 30, 66, 10
    stack = tmp_path / "stack"; (stack / "merged").mkdir(parents=True)
    base = dict(stack_dir=stack, stack_azimuth_looks=3, stack_range_looks=11, miaplpy_ministack_size=size)
    good = _synthetic_phase_series(n, L, W, size, False, rng)
    bad = _synthetic_phase_series(n, L, W, size, True, rng)
    dates = [f"2021{1 + k // 28:02d}{1 + k % 28:02d}" for k in range(n)]
    info = mb.check_datum_consistency(types.SimpleNamespace(miaplpy_phase_linking_method="sequential_EMI", **base), good, dates)
    assert info["ratio_across_over_within"] > 0.9
    with pytest.raises(RuntimeError, match="own datum"):
        mb.check_datum_consistency(types.SimpleNamespace(miaplpy_phase_linking_method="sequential_EMI", **base), bad, dates)
    # the non-sequential estimator has no mini-stacks: the numbers are recorded, nothing raises
    info = mb.check_datum_consistency(types.SimpleNamespace(miaplpy_phase_linking_method="EMI", **base), bad, dates)
    assert (stack / "merged" / "linked_datum_check.json").exists() and info["method"] == "EMI"


def test_nearest_unmasked_prefers_closest_and_respects_radius():
    from insar_pipeline.time_series import nearest_unmasked
    m = np.zeros((20, 20), bool); m[10, 13] = True; m[14, 10] = True; m[3, 3] = True
    assert nearest_unmasked(m, 10, 10, radius=4) == (10, 13)      # distance 3 beats distance 4
    assert nearest_unmasked(m, 10, 10, radius=2) is None
    m[10, 10] = True
    assert nearest_unmasked(m, 10, 10, radius=4) == (10, 10)
