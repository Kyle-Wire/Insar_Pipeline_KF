"""PLAN phases 2 and 3, finishing pass (review 8 Oct 2026): the pieces the
seven groups wrote, wired together end to end.

What is checked:
  - config.py declares the settings the groups left undeclared: the NISAR
    route's nisar_* (defaults = nisar_backend.SETTINGS), the plate-fixed
    GNSS datum's reference_gnss_frame / reference_gnss_plate (defaults =
    significance.REFERENCE_FRAME_SETTINGS) and validation_points; a YAML may
    set them, and bad values are refused in words;
  - every experiment config (Sentinel-1, DISP-S1, NISAR) is one run of the
    protocol it names, through runs.<id>.config or a dated amendment's
    run_configs (the NISAR configs through 2026-10-08/nisar-configs);
  - the A configs use the digitised T1 files: the deramp polygon (recorded
    by protocol A's amendment 2026-10-08/a-deramp-polygon, which the harness
    checks), the panel outlines and survey lines (event_polygons /
    event_lines) and, for A1/A3, the Tahmoor West units (validation_points);
    the event products run on them;
  - the harness reads the config's validation_points when no --points is
    given, and warns when a run's deramp polygon is not the recorded one;
  - NISAR: the harness's reliability quantity is the mean pair coherence of
    nisar_gunw/coherence_gate.tif (protocol_B1), deliverables.geometry_field
    reads MintPy's mintpy/inputs/geometryGeo.h5 (and gives the left-looking
    line of sight), and `insar cleanup` offers nisar_gunw/;
  - `insar validate`'s usage line lists --outlines and --lines.
Each test fails on the tree before the finishing pass. No network, nothing
under ~/insar_projects (INSAR_PROJECTS -> tmp_path).
"""
from __future__ import annotations

import dataclasses
import math
import os
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
DEFORM = EXP / "context" / "appin_tahmoor_deforming.geojson"

# every experiment config -> (protocol, run)
RUN_OF = {
    "exp_a1_appin_des147": ("A", "A1"), "exp_a2_appin_disp_f39360": ("A", "A2"), "exp_a3_appin_asc9": ("A", "A3"),
    "exp_a5a_appin_nisar_asc167": ("A", "A5"), "exp_a5b_appin_nisar_asc95": ("A", "A5"),
    "exp_a5c_appin_nisar_des16": ("A", "A5"),
    "exp_b1_corcoran_asc137": ("B1", "B1"), "exp_b2_corcoran_disp_f36542": ("B1", "B2"),
    "exp_b3_corcoran_des144": ("B1", "B3"), "exp_b4_corcoran_disp_f38502": ("B1", "B4"),
    "exp_b5a_corcoran_nisar_asc34": ("B1", "B5"), "exp_b5b_corcoran_nisar_des114": ("B1", "B5"),
    "exp_b5c_elnido_nisar_asc34": ("B1", "B5"), "exp_b5d_elnido_nisar_des114": ("B1", "B5"),
    "exp_b5e_elnido_nisar_des42": ("B1", "B5"),
    "exp_b2_houston_asc34": ("B2", "B-2a"), "exp_b2b_houston_disp_f08882": ("B2", "B-2b"),
    "exp_b2c_houston_nisar_asc62": ("B2", "B-2c"),
}


def _yaml(tmp_path, extra=""):
    f = tmp_path / "c.yaml"
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-119.82, 36.08, -119.52, 36.35]\n"
                 "start_date: '2025-11-01'\nend_date: '2026-09-30'\nbackend: nisar\ntrack: 34\n"
                 "orbit_direction: ascending\n" + extra)
    return f


# ====================================================================== settings

def test_new_settings_are_config_fields_with_their_modules_defaults():
    from insar_pipeline import config, nisar_backend, significance
    fields = {f.name: f.default for f in dataclasses.fields(config.PipelineConfig)}
    for key, default in nisar_backend.SETTINGS.items():
        assert key in fields and fields[key] == default, key
    for key, (default, _meaning) in significance.REFERENCE_FRAME_SETTINGS.items():
        assert key in fields and fields[key] == default, key
    assert fields["validation_points"] == ""


def test_a_yaml_may_set_them_and_bad_values_are_refused(tmp_path):
    from insar_pipeline.config import ConfigError, load_config
    cfg = load_config(str(_yaml(tmp_path, "nisar_ionosphere: 'no'\nnisar_chain_gap: fail\nnisar_polarization: vv\n"
                                          "reference_gnss_frame: itrf\nreference_gnss_plate: NOAM\n")))
    assert (cfg.nisar_ionosphere, cfg.nisar_chain_gap, cfg.reference_gnss_frame) == ("no", "fail", "itrf")
    from insar_pipeline import nisar_backend
    assert nisar_backend.setting(cfg, "nisar_chain_gap") == "fail"
    for bad, words in (("nisar_fetch: ftp\n", "nisar_fetch 'ftp' is not one of"),
                       ("nisar_polarization: HV\n", "nisar_polarization"),
                       ("nisar_min_valid_ratio: 1.5\n", "between 0 and 1"),
                       ("nisar_posting_m: 0\n", "above 0")):
        with pytest.raises(ConfigError, match=words):
            load_config(str(_yaml(tmp_path, bad)))


# ====================================================================== configs and protocols

# ====================================================================== the harness

def test_harness_reads_the_configs_validation_points(tmp_path):
    dates = H._dates()
    cfg = H.make_product(tmp_path / "proj", dates)
    cfg.validation_points = "units.csv"
    gt = H.FakeGT(H.STATIONS_A)
    gt.points = H._unit_points(["20211215", "20220601", "20230110", "20240501"])
    res = V.validate(cfg, H.a_protocol(tmp_path), tmp_path / "out", run="T1", cache_dir=tmp_path / "c", gt=gt,
                     sensitivity=False, epoch_coverage=False)
    assert ("points", "units.csv") in gt.calls
    units = res["rows"]["tahmoor_w_units"]
    assert units["grade"] != "not_computed" and units["metrics"]["n"] >= 1
    assert res["truth"]["points_file"]["from"] == "the config's validation_points"
    # --points wins over the config
    gt2 = H.FakeGT(H.STATIONS_A)
    gt2.points = gt.points
    (tmp_path / "x").mkdir()
    res2 = V.validate(cfg, H.a_protocol(tmp_path / "x"), tmp_path / "out2", run="T1", cache_dir=tmp_path / "c2",
                      gt=gt2, sensitivity=False, epoch_coverage=False, points="other.csv")
    assert ("points", "other.csv") in gt2.calls and res2["truth"]["points_file"]["from"] == "--points"


def test_a_run_whose_deramp_polygon_is_not_the_recorded_one_is_said_so(tmp_path):
    P = yaml.safe_load(H.b1_protocol(tmp_path).read_text())
    P["amendments"] = [{"id": "2026-10-08/x", "kind": "record only", "before_any_result": True,
                        "harness": {"run_deramp_polygon": {"T1": "configs/experiments/context/appin_tahmoor_deforming.geojson"}}}]
    proto = tmp_path / "protocol_x.yaml"
    proto.write_text(yaml.safe_dump(P, sort_keys=False))
    dates = H._dates(step=24)
    cfg = H.make_product(tmp_path / "p", dates)
    kw = dict(run="T1", gt=H.FakeGT(H.STATIONS_B), sensitivity=False, epoch_coverage=False)
    cfg.deramp_exclude_geojson = "/somewhere/appin_tahmoor_deforming_APPROX.geojson"
    res = V.validate(cfg, proto, tmp_path / "o1", cache_dir=tmp_path / "c1", **kw)
    assert any("deramp polygon is recorded as" in w for w in res["warnings"])
    cfg.deramp_exclude_geojson = str(DEFORM)
    res = V.validate(cfg, proto, tmp_path / "o2", cache_dir=tmp_path / "c2", **kw)
    assert not any("deramp polygon is recorded as" in w for w in res["warnings"])


def _gate(cfg, mean):
    """nisar_gunw/coherence_gate.tif as nisar_backend writes it: on a lattice one cell wider on every side."""
    import rasterio
    from rasterio.transform import from_origin
    p = Path(cfg.project_dir) / "nisar_gunw" / "coherence_gate.tif"
    p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(p, "w", driver="GTiff", height=mean.shape[0], width=mean.shape[1], count=2, dtype="float32",
                       crs="EPSG:4326", nodata=np.nan,
                       transform=from_origin(H.LON0 - H.DLON, H.LAT0 - H.DLAT, H.DLON, -H.DLAT)) as dst:
        dst.write(mean.astype(np.float32), 1)
        dst.write((mean >= 0.4).astype(np.float32), 2)
    return p


def test_nisar_reliability_is_the_mean_pair_coherence(tmp_path):
    """protocol_B1: 'NISAR GUNW: mean interferometric coherence over the pairs
    used >= 0.4 (temporal coherence is 1.0 by construction on a chain)'."""
    dates = H._dates(step=24)
    cfg = H.make_product(tmp_path / "p", dates, tcoh=np.ones((H.NY, H.NX)))
    cfg.backend = "nisar"
    prod = V.InsarProduct(cfg)                                    # no gate yet: said so, every cell reliable
    assert "missing" in prod.reliability and np.all(prod.tcoh == 1.0)
    mean = np.full((H.NY + 2, H.NX + 2), 0.8)
    mean[:, : H.NX // 2 + 1] = 0.3                                # the west half of the product (+ margin)
    _gate(cfg, mean)
    prod = V.InsarProduct(cfg)
    assert prod.reliability.startswith("mean interferometric coherence")
    assert np.allclose(prod.tcoh[:, : H.NX // 2], 0.3) and np.allclose(prod.tcoh[:, H.NX // 2:], 0.8)
    assert prod.describe()["reliability_quantity"] == prod.reliability
    # a Sentinel-1 run never reads it
    cfg.backend = "miaplpy"
    assert np.all(V.InsarProduct(cfg).tcoh == 1.0)


def test_geometry_field_reads_the_nisar_routes_mintpy_geometry(tmp_path):
    import h5py

    from insar_pipeline import deliverables as dl
    cfg = SimpleNamespace(project_dir=tmp_path, mintpy_dir=tmp_path / "mintpy", backend="nisar")
    (cfg.mintpy_dir / "inputs").mkdir(parents=True)
    with h5py.File(cfg.mintpy_dir / "inputs" / "geometryGeo.h5", "w") as f:
        f.create_dataset("incidenceAngle", data=np.full((4, 5), 41.8, np.float32))
        f.create_dataset("azimuthAngle", data=np.full((4, 5), -74.13, np.float32))   # left-looking ascending
        f.create_dataset("height", data=np.zeros((4, 5), np.float32))
    e, n_, u = (dl.geometry_field(cfg, k) for k in ("los_east", "los_north", "los_up"))
    assert e[0, 0] == pytest.approx(0.6411, abs=1e-3) and n_[0, 0] == pytest.approx(0.1822, abs=1e-3)
    assert u[0, 0] == pytest.approx(math.cos(math.radians(41.8)), abs=1e-4)
    assert dl.geometry_field(cfg, "incidence")[0, 0] == pytest.approx(41.8)
    # only on the NISAR route: another route that lacks its own geometry reads what it did (nothing)
    assert dl.geometry_field(SimpleNamespace(**{**vars(cfg), "backend": "hyp3"}), "incidence") is None
    # another route's geocoded geometry still wins: the inputs file is read only when there is none
    (cfg.mintpy_dir / "geo").mkdir()
    with h5py.File(cfg.mintpy_dir / "geo" / "geo_geometryRadar.h5", "w") as f:
        f.create_dataset("incidenceAngle", data=np.full((4, 5), 35.0, np.float32))
        f.create_dataset("azimuthAngle", data=np.full((4, 5), 100.0, np.float32))
    assert dl.geometry_field(cfg, "incidence")[0, 0] == pytest.approx(35.0)


def test_insar_cleanup_offers_the_nisar_folder(tmp_path):
    from insar_pipeline.config import load_config
    from insar_pipeline.scripts import cleanup
    cfg = load_config(str(_yaml(tmp_path)))
    P = Path(cfg.project_dir) / "nisar_gunw"
    (P / "crops").mkdir(parents=True)
    (P / "stack").mkdir()
    (P / "G_whole.h5").write_bytes(b"x" * 100)
    for nm in ("NISAR_GUNW_CROP_T034A_1_2.h5", "NISAR_GUNW_CROP_T034A_9_10.h5"):
        (P / "crops" / nm).write_bytes(b"y" * 10)
    (P / "stack" / "NISAR_GUNW_CROP_T034A_1_2.h5").symlink_to(P / "crops" / "NISAR_GUNW_CROP_T034A_1_2.h5")
    got = {(lv, p.name) for lv, p, _ in cleanup.targets(cfg, 2)}
    assert got == {(1, "G_whole.h5"), (1, "NISAR_GUNW_CROP_T034A_9_10.h5")}     # level 2 waits for MintPy's load
    (Path(cfg.mintpy_dir) / "inputs").mkdir(parents=True)
    (Path(cfg.mintpy_dir) / "inputs" / "ifgramStack.h5").write_bytes(b"s")
    got = {(lv, p.name) for lv, p, _ in cleanup.targets(cfg, 2)}
    assert {(2, "crops"), (2, "stack")} <= got
    assert {p.name for lv, p, _ in cleanup.targets(cfg, 1)} == {"G_whole.h5", "NISAR_GUNW_CROP_T034A_9_10.h5"}


def test_validate_usage_lists_the_t1_options(capsys):
    from insar_pipeline import cli
    assert cli.validate_main([]) == 2
    out = capsys.readouterr()
    text = out.out + out.err
    assert "--outlines FILE" in text and "--lines FILE" in text and "--points FILE" in text


def test_readme_lists_every_run_with_its_launch_line():
    text = (EXP / "README.md").read_text()
    for name in RUN_OF:
        assert f"insar ~/insar_pipeline/configs/experiments/{name}.yaml" in text, name
    for sha in (V._sha256(REVIEW / f"protocol_{n}.yaml") for n in ("A", "B1", "B2")):
        assert sha in text
    assert "2026-10-08/nisar-configs" in text and "2026-10-08/a-deramp-polygon" in text
    assert os.path.basename(str(DEFORM)) in text
