"""Lane L4a (config, machine settings, doctor) - review of 5 Oct 2026.

Each test fails on the code before the fixes and passes after. Synthetic data
only; no network, nothing under ~/insar_projects or ~/.config/insar touched.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import stat
from collections import namedtuple
from pathlib import Path

import pytest
import yaml

from insar_pipeline import config as C
from insar_pipeline import doctor, settings
from insar_pipeline.config import ConfigError, PipelineConfig, load_config, validate_config

DU = namedtuple("usage", "total used free")


# ---------------------------------------------------------------- helpers
def _poly(path: Path, w, s, e, n) -> Path:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                    "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}]}))
    return path


def _base(tmp_path: Path) -> dict:
    """A clean phase-linked config like the builder writes for a new site."""
    ctx = tmp_path / "proj" / "context"
    ctx.mkdir(parents=True, exist_ok=True)
    return {
        "backend": "miaplpy", "project_dir": str(tmp_path / "proj"),
        "bbox": [-87.457, 38.881, -87.203, 39.079], "start_date": "2016-01-01", "end_date": "2026-09-30",
        "track": 165, "orbit_direction": "descending",
        "report_areas": ["Full Satellite Imaging:full",
                         f"Zone:{_poly(ctx / 'zone.geojson', -87.40, 38.93, -87.26, 39.03)}",
                         "Pit 3:38.99,-87.35,39.0,-87.34"],
        "surface_mines_geojson": str(_poly(ctx / "surface_mines.geojson", -87.36, 38.98, -87.33, 39.01)),
        "mintpy_deramp": "linear",
        "deramp_exclude_geojson": str(_poly(ctx / "deforming_area.geojson", -87.36, 38.98, -87.33, 39.01)),
        "stack_swaths": "2", "stack_unw_engine": "snaphu2", "stack_unw_cost_mode": "DEFO",
        "stack_virtual_merge": "False", "stack_cleanup_coreg_after_filter": False,
        "miaplpy_phase_linking_method": "sequential_EMI", "miaplpy_ministack_size": 10,
    }


def _write(tmp_path: Path, doc: dict, name="c.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc))
    return p


def _cfg(tmp_path, **over) -> PipelineConfig:
    kw = dict(project_dir=tmp_path / "p", bbox=[-87.8, 38.2, -87.52, 38.47],
              start_date="2018-06-01", end_date="2019-06-30", backend="stack")
    kw.update(over)
    return PipelineConfig(**kw)


# ---------------------------------------------------------------- usability-2: real disk budget
def _fake_wsl(monkeypatch, tmp_path, fs_free=743e9, used=283e9, c_free=150e9, vhdx_bytes=475e9):
    """WSL with a 475 GB vhdx (sparse file: costs no disk) on a fake drive."""
    drive = tmp_path / "mnt_c"
    vh = drive / "Users" / "u" / "AppData" / "Local" / "wsl" / "x" / "ext4.vhdx"
    vh.parent.mkdir(parents=True)
    with open(vh, "wb") as f:
        f.truncate(int(vhdx_bytes))
    host = str(Path("/", *vh.parts[1:3]))           # what disk_budget_gb takes as the drive

    def usage(p):
        p = str(p)
        if p == "/":
            return DU(1081e9, used, fs_free)
        if p == host:
            return DU(999e9, 999e9 - c_free, c_free)
        return DU(1081e9, used, fs_free)
    monkeypatch.setattr(settings, "_is_wsl", lambda: True)
    monkeypatch.setattr(settings, "_mount_kind", lambda p: "root")
    monkeypatch.setattr(settings, "_vhdx_path", lambda: (vh, "test"))
    monkeypatch.setattr(settings, "_usage", usage)
    # P0.2 (8 Oct 2026): the budget counts what the vhdx HOLDS on the drive,
    # not its length. These tests are the non-sparse case (held = length);
    # the sparse case is tests/test_p0_disk.py.
    monkeypatch.setattr(settings, "_vhdx_sizes", lambda v: (vhdx_bytes, vhdx_bytes), raising=False)
    return vh


def test_disk_budget_counts_vhdx_headroom_plus_windows_free(tmp_path, monkeypatch):
    _fake_wsl(monkeypatch, tmp_path)
    b = settings.disk_budget_gb(tmp_path / "projects" / "not_yet")
    # (475 - 283) + 150 = 342, not the 743 ext4 reports
    assert b["budget"] == pytest.approx(342, abs=0.2)
    assert b["fs_free"] == pytest.approx(743, abs=0.2) and b["budget"] < b["fs_free"]
    assert b["vhdx_gb"] == pytest.approx(475, abs=0.2) and b["c_free"] == pytest.approx(150, abs=0.2)
    assert set(b) >= {"fs_free", "vhdx_gb", "used_gb", "c_free", "budget", "source"}


def test_disk_budget_is_capped_by_the_vhdx_virtual_size(tmp_path, monkeypatch):
    _fake_wsl(monkeypatch, tmp_path, fs_free=100e9)
    assert settings.disk_budget_gb(tmp_path)["budget"] == pytest.approx(100, abs=0.2)


def test_disk_budget_off_wsl_or_without_vhdx_is_filesystem_free_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_usage", lambda p: DU(1000e9, 200e9, 500e9))
    monkeypatch.setattr(settings, "_is_wsl", lambda: False)
    b = settings.disk_budget_gb(tmp_path)
    assert b["budget"] == 500 and b["vhdx_gb"] is None
    monkeypatch.setattr(settings, "_is_wsl", lambda: True)
    monkeypatch.setattr(settings, "_mount_kind", lambda p: "root")
    monkeypatch.setattr(settings, "_vhdx_path", lambda: (None, "none here"))
    b = settings.disk_budget_gb(tmp_path)
    assert b["budget"] == 500 and "vhdx not found" in b["source"]
    monkeypatch.setattr(settings, "_mount_kind", lambda p: "windows")
    assert "Windows drive" in settings.disk_budget_gb(tmp_path)["source"]


def test_doctor_warns_on_the_real_budget_not_ext4_free(tmp_path, monkeypatch):
    _fake_wsl(monkeypatch, tmp_path)
    lines = doctor.disk_checks(tmp_path)
    room = next(x for x in lines if "phase-linked run" in x[1])
    assert room[0] is False and room[3] is True              # 342 < 350: a warning
    assert "342" in lines[0][1] and "743" in lines[0][1]     # both shown, the budget first
    _fake_wsl(monkeypatch, tmp_path / "big", c_free=400e9)
    assert next(x for x in doctor.disk_checks(tmp_path) if "phase-linked" in x[1])[0] is True


# ---------------------------------------------------------------- usability-3 / generality-13: RAM-sized pools
def test_sized_parallelism_reproduces_todays_values_on_the_measured_machine():
    want = {"stack_num_proc": 12, "stack_num_proc_topo": 3, "stack_num_proc_unwrap": 24,
            "mintpy_num_workers": 16, "mintpy_max_memory_gb": 16.0, "miaplpy_num_processors": 20}
    assert settings.sized_parallelism(23, 24) == want
    assert settings.sized_parallelism(23.47, 24) == want       # MemTotal of a 24GB .wslconfig
    # every key is a config field, so the dict can go straight into a YAML
    assert set(want) <= {f for f in PipelineConfig.__dataclass_fields__}


def test_sized_parallelism_fits_a_16_gb_laptop():
    p = settings.sized_parallelism(7.8, 8)                     # WSL default on a 16 GB PC
    assert p["mintpy_max_memory_gb"] <= 5 and p["mintpy_num_workers"] <= 8
    assert p["stack_num_proc"] * 1.27 <= 0.8 * 7.8             # 1.27 GB peak RSS per merge (measured)
    assert p["stack_num_proc_unwrap"] * 0.53 <= 0.8 * 7.8      # 0.53 GB per snaphu (measured)
    assert all(v >= 1 for v in settings.sized_parallelism(1.0, 1).values())


def test_machine_resources_reads_meminfo(tmp_path, monkeypatch):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:        8178892 kB\nMemFree:  100 kB\nMemAvailable:    6000000 kB\nSwapTotal:  2097152 kB\n")
    monkeypatch.setattr(settings, "_MEMINFO", f)
    r = settings.machine_resources()
    assert r["mem_gb"] == pytest.approx(7.8, abs=0.01) and r["swap_gb"] == pytest.approx(2.0, abs=0.01)
    assert r["cores"] >= 1


def test_config_pool_defaults_are_sized_for_the_machine_and_yaml_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "machine_resources", lambda: {"mem_gb": 7.8, "swap_gb": 0, "cores": 8})
    c = _cfg(tmp_path)
    assert c.stack_num_proc == 2 and c.stack_num_proc_unwrap == 5 and c.mintpy_max_memory_gb == 5.0
    assert _cfg(tmp_path, stack_num_proc_unwrap=24).stack_num_proc_unwrap == 24
    monkeypatch.setattr(settings, "machine_resources", lambda: {"mem_gb": 23.47, "swap_gb": 16, "cores": 24})
    c = _cfg(tmp_path)
    assert (c.stack_num_proc, c.stack_num_proc_topo, c.stack_num_proc_unwrap, c.mintpy_max_memory_gb) == (12, 3, 24, 16.0)


def test_doctor_memory_check_gives_the_wslconfig_fix():
    small = doctor.memory_checks({"mem_gb": 7.8, "swap_gb": 2, "cores": 8}, wsl=True)
    assert small[0][0] is False and small[0][3] is False                 # < 12 GB: a problem
    assert ".wslconfig" in small[0][2] and "wsl --shutdown" in small[0][2] and "memory=" in small[0][2]
    mid = doctor.memory_checks({"mem_gb": 15.9, "swap_gb": 4, "cores": 16}, wsl=True)
    assert mid[0][0] is False and mid[0][3] is True                      # < 20 GB: a warning
    assert doctor.memory_checks({"mem_gb": 23.5, "swap_gb": 16, "cores": 24}, wsl=True)[0][0] is True


def test_doctor_flags_configs_with_pools_bigger_than_the_machine(tmp_path):
    (tmp_path / "copied.yaml").write_text("stack_num_proc_unwrap: 24\nmintpy_max_memory_gb: 16\n")
    (tmp_path / "fine.yaml").write_text("stack_num_proc_unwrap: 2\n")
    out = doctor.pool_checks(tmp_path, {"mem_gb": 7.8, "swap_gb": 0, "cores": 8})
    assert len(out) == 1 and "copied.yaml" in out[0][1] and "stack_num_proc_unwrap 24" in out[0][1]
    assert doctor.pool_checks(tmp_path, {"mem_gb": 23.5, "swap_gb": 16, "cores": 24}) == []


# ---------------------------------------------------------------- phaselinked-10 / timeseries-7: rejected reference
def _ref_project(tmp_path, choice: dict, yaml_ref="38.950000,-87.300000"):
    doc = _base(tmp_path)
    doc["reference_lalo"] = yaml_ref
    (tmp_path / "proj" / "stack").mkdir(parents=True, exist_ok=True)
    (tmp_path / "proj" / "stack" / "reference_choice.json").write_text(json.dumps(choice))
    return _write(tmp_path, doc)


def test_rejected_yaml_reference_is_replaced_by_the_runs_choice(tmp_path):
    p = _ref_project(tmp_path, {"configured": {"lat": 38.95, "lon": -87.3, "fraction_of_pairs_unwrapped": 0.7},
                                "pairs": 800, "kept_configured": False,
                                "chosen": {"lat": 39.01234567, "lon": -87.41234567}})
    c = load_config(str(p))
    assert c.reference_lalo == "39.012346,-87.412346"
    assert c.reference_auto is True and c.reference_yaml_rejected == "38.950000,-87.300000"


def test_yaml_reference_wins_when_kept_or_changed_after_the_run(tmp_path):
    kept = {"configured": {"lat": 38.95, "lon": -87.3}, "kept_configured": True,
            "chosen": {"lat": 38.95, "lon": -87.3}}
    assert load_config(str(_ref_project(tmp_path, kept))).reference_lalo == "38.950000,-87.300000"
    # the user set a NEW reference after the run rejected the old one: theirs wins
    moved = {"configured": {"lat": 38.90, "lon": -87.25}, "kept_configured": False,
             "chosen": {"lat": 39.0, "lon": -87.4}}
    c = load_config(str(_ref_project(tmp_path, moved)))
    assert c.reference_lalo == "38.950000,-87.300000" and not getattr(c, "reference_auto", False)
    # blank YAML still takes the run's choice (the case fixed earlier)
    blank = _ref_project(tmp_path, {"configured": None, "chosen": {"lat": 39.0, "lon": -87.4}}, yaml_ref="")
    assert load_config(str(blank)).reference_lalo == "39.000000,-87.400000"


# ---------------------------------------------------------------- phaselinked-11: MiaplPy settings
def test_miaplpy_settings_are_validated(tmp_path):
    ok = _cfg(tmp_path, backend="miaplpy", stack_virtual_merge="False")
    assert validate_config(ok) == []
    for over, word in ((dict(miaplpy_time_lag=0), "miaplpy_time_lag"),           # measured: datum defect back
                       (dict(miaplpy_time_lag=10), "miaplpy_time_lag"),          # overlap = whole window
                       (dict(miaplpy_ministack_size=1, miaplpy_time_lag=1), "miaplpy_ministack_size"),
                       (dict(miaplpy_phase_linking_method="sequential_emi"), "miaplpy_phase_linking_method"),
                       (dict(miaplpy_network_connections=-1), "miaplpy_network_connections"),
                       (dict(miaplpy_pair_coherence="phase"), "miaplpy_pair_coherence"),
                       (dict(stack_virtual_merge="True"), "stack_virtual_merge"),
                       (dict(stack_cleanup_coreg_after_filter=True), "stack_cleanup_coreg_after_filter")):
        probs = validate_config(_cfg(tmp_path, **{"backend": "miaplpy", "stack_virtual_merge": "False", **over}))
        assert any(word in p for p in probs), (over, probs)
    assert validate_config(_cfg(tmp_path, backend="miaplpy", stack_virtual_merge="False", miaplpy_time_lag=1)) == []


def test_ministack_longer_than_the_record_is_refused():
    c = PipelineConfig(project_dir=Path("/tmp/x"), bbox=[0, 0, 1, 1], start_date="2020-01-01",
                       end_date="2020-06-01", backend="miaplpy", miaplpy_ministack_size=10)
    assert "fewer than one mini-stack" in C.miaplpy_stack_problem(c, 8)
    assert C.miaplpy_stack_problem(c, 10) is None


# ---------------------------------------------------------------- generality-6: the dangerous-config battery
BATTERY = {
    "start after end": {"start_date": "2026-01-01", "end_date": "2016-01-01"},
    "date format 2016/01/01": {"start_date": "2016/01/01"},
    "track null (miaplpy)": {"track": None},
    "report area without colon": {"report_areas": ["Full Satellite Imaging:full", "Pit 3 38.99,-87.35,39.0,-87.34"]},
    "colon inside a report-area name": {"report_areas": ["Pit 3: north:38.99,-87.35,39.0,-87.34"]},
    "report area geojson missing": {"report_areas": ["Full Satellite Imaging:full", "Pit:/nonexistent/pit.geojson"]},
    "report area outside bbox": {"report_areas": ["Full Satellite Imaging:full", "Elsewhere:40.0,-86.0,40.1,-85.9"]},
    "report area name used twice": {"report_areas": ["Pit:38.99,-87.35,39.0,-87.34", "pit:38.95,-87.35,38.96,-87.34"]},
    "stack_swaths 'IW2'": {"stack_swaths": "IW2"},
    "deramp linear, polygon missing": {"deramp_exclude_geojson": "/nonexistent/mines.geojson"},
    "deramp linear, no polygon at all": {"deramp_exclude_geojson": ""},
    "surface_mines_geojson missing": {"surface_mines_geojson": "/nonexistent/s.geojson"},
    "reference outside bbox": {"reference_lalo": "40.0,-86.0"},
    "reference given lon,lat": {"reference_lalo": "-87.3,38.95"},
    "stack_reference_date 2021-02-19": {"stack_reference_date": "2021-02-19"},
    "stack_reference_date outside the dates": {"stack_reference_date": "20150101"},
    "focus site malformed": {"focus_sites_lalo": ["Parcel 38.99 -87.35"]},
    "focus site 'full'": {"focus_sites_lalo": ["Parcel:full"]},
    "stack_unw_engine typo": {"stack_unw_engine": "snaphu-2"},
    "cost mode typo": {"stack_unw_cost_mode": "DEFORM"},
    "unwrap-error method typo": {"mintpy_unwrap_error_method": "bridging+closure"},
    "miaplpy + virtual merge True": {"stack_virtual_merge": "True"},
    "gnss_tie_radius_km -5": {"gnss_tie_radius_km": -5},
    "far-field buffer -100": {"acceptance_far_field_buffer_m": -100},
    "deramp buffer -1": {"deramp_buffer_m": -1},
    "stable_area_bbox wrong order": {"stable_area_bbox": [38.9, -87.4, 39.0, -87.3]},
    "bbox 5 deg wide": {"bbox": [-90.0, 37.0, -85.0, 40.0]},
    "miaplpy_ministack_size 0": {"miaplpy_ministack_size": 0},
    "end_date in the future 2030": {"end_date": "2030-01-01"},
    "min_unwrap_consistency 1.5": {"min_unwrap_consistency": 1.5},
    "allow_missing_dates bad date": {"allow_missing_dates": ["2021-02-19"]},
    "mined_year_geojson without its field": {"mined_year_geojson": "__ZONE__"},
    "landcover raster missing": {"landcover_raster": "/nonexistent/nlcd.tif"},
    "report polygon outside bbox": {"report_areas": ["Full Satellite Imaging:full", "Far:__FAR__"]},
    # two labels whose slugs collided wrote one file for both (collide.py)
    "two report areas share one polygon file": {"report_areas": ["Pit 3:__ZONE__", "Pit-3:__ZONE__"]},
}


def test_the_clean_base_config_loads(tmp_path):
    c = load_config(str(_write(tmp_path, _base(tmp_path))))
    assert c.backend == "miaplpy" and c.track == 165
    # display units in mm is a legal choice, not a mistake
    d = _base(tmp_path); d["display_units"] = "mm"
    assert load_config(str(_write(tmp_path, d))).display_units == "mm"


@pytest.mark.parametrize("case", sorted(BATTERY))
def test_dangerous_configs_are_rejected_at_load(tmp_path, case):
    d = copy.deepcopy(_base(tmp_path))
    zone = str(tmp_path / "proj" / "context" / "zone.geojson")
    far = str(_poly(tmp_path / "proj" / "context" / "far.geojson", -86.0, 40.0, -85.9, 40.1))
    sub = lambda v: zone if v == "__ZONE__" else (v.replace("__ZONE__", zone).replace("__FAR__", far)  # noqa: E731
                                                 if isinstance(v, str) else v)
    upd = {k: ([sub(x) for x in v] if isinstance(v, list) else sub(v)) for k, v in BATTERY[case].items()}
    d.update(upd)
    with pytest.raises(ConfigError):
        load_config(str(_write(tmp_path, d)))


def test_unquoted_yaml_dates_still_load_as_strings(tmp_path):
    d = _base(tmp_path)
    text = yaml.safe_dump(d).replace("'2016-01-01'", "2016-01-01")
    text += "stack_reference_date: 20210219\n"
    p = tmp_path / "u.yaml"; p.write_text(text)
    c = load_config(str(p))
    assert c.start_date == "2016-01-01" and c.stack_reference_date == "20210219"


def test_every_existing_config_still_loads():
    pkg = Path(C.__file__).resolve().parent
    files = sorted((pkg / "configs").glob("*.yaml")) + sorted((pkg / "configs" / "archive").glob("*.yaml"))
    fr = Path.home() / "insar_projects" / "configs" / "site_b.yaml"
    files += [fr] if fr.exists() else []
    if not files:
        # the example configs left the repository on 28 Oct 2026: a fresh clone has none
        pytest.skip("no project configs on this machine")
    root = str(settings.projects_root())
    for f in files:
        try:
            load_config(str(f))
        except ConfigError as exc:
            # on a machine without Kyle's projects the only acceptable problem
            # is an input that lives in one of those projects
            lines = [ln.strip() for ln in str(exc).splitlines()[1:]]
            assert lines and all("does not exist" in ln and root in ln for ln in lines), (f, lines)


def test_new_contract_fields_and_route_tags():
    c = PipelineConfig(project_dir=Path("/tmp/x"), bbox=[0, 0, 1, 1], start_date="2020-01-01", end_date="2020-06-01")
    assert c.gnss_tie_radius_km == C.GNSS_TIE_RADIUS_KM == 40.0
    assert (c.disp_client_report, c.points_full_csv, c.full_timeseries_kmz, c.cross_check_disp,
            c.injection_test) == (False,) * 5
    assert c.min_unwrap_consistency == 0.0 and c.deramp_buffer_m == 2000.0
    assert c.control_ground_label == "farmland" and c.allow_missing_dates == () and c.mine_layer_sources == ()
    assert C.route_tag(c) == "comparison" and not C.is_screening(c)
    c.backend = "disp"
    assert C.is_screening(c) and C.route_tag(c) == "screening"
    c.backend = "miaplpy"
    assert C.route_tag(c) == "official"


def test_identity_comes_from_the_settings_file(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"firm": "Acme Geo", "preparer": "K. W."}))
    monkeypatch.setenv("INSAR_SETTINGS", str(f))
    assert settings.identity() == {"firm": "Acme Geo", "preparer": "K. W.", "contact": ""}
    monkeypatch.setenv("INSAR_SETTINGS", str(tmp_path / "none.json"))
    assert settings.identity()["firm"] == ""


# ---------------------------------------------------------------- tests-11: doctor runs the suite + known answers
def test_doctor_suite_check_turns_red_on_a_failing_test(tmp_path):
    t = tmp_path / "pkg" / "tests"
    t.mkdir(parents=True)
    (t / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    ok, summary = doctor.suite_check(t, timeout=120)
    assert ok and "passed" in summary
    (t / "test_bad.py").write_text("def test_bad():\n    assert 0.1 + 0.2 == 0.3\n")
    ok, summary = doctor.suite_check(t, timeout=120)
    assert not ok and "failed" in summary


def test_snaphu_known_answer_catches_a_wrong_unwrapper(tmp_path):
    fake = tmp_path / "snaphu"
    # a "snaphu" that runs, exits 0 and writes zeros: a version check would pass it
    fake.write_text("#!/usr/bin/env python3\nimport sys\nconf = open(sys.argv[2]).read().split()\n"
                    "out = conf[conf.index('OUTFILE') + 1]\nw = int(sys.argv[4])\n"
                    "open(out, 'wb').write(bytes(4 * w * 64))\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    ok, detail = doctor.snaphu_known_answer(str(fake))
    assert not ok and "residual" in detail
    try:
        from insar_pipeline.unwrap_crop import find_snaphu
        real = find_snaphu("")
    except Exception:                                          # noqa: BLE001
        real = shutil.which("snaphu")
    if real and os.access(real, os.X_OK):
        ok, detail = doctor.snaphu_known_answer(real)
        assert ok, detail


# ================================================================ lane L4b
# The builder (areas.py, map_builder.py, project_builder.html). Synthetic
# outlines; the burst catalogue is faked; INSAR_PROJECTS / INSAR_SETTINGS point
# at tmp_path, so nothing under ~/insar_projects or ~/.config/insar is touched.
from insar_pipeline import areas  # noqa: E402

PKG_DIR = Path(__file__).resolve().parents[1]
HTML = (PKG_DIR / "project_builder.html").read_text()


def _cover(calls=None, covered=1.0, swaths=(2,), fail=None):
    def fake(area, track, direction, targets=None, **k):
        if calls is not None:
            calls.append((area.bbox, int(track), direction))
        if fail:
            raise fail
        return {"swaths": list(swaths), "bursts": 3, "covered": covered,
                "targets": {n: covered for n in (targets or {})}}
    return fake


@pytest.fixture
def builder_env(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("INSAR_PROJECTS", str(root))
    monkeypatch.setenv("INSAR_SETTINGS", str(tmp_path / "settings.json"))
    # raising=False: the same tests run against the old code, which had no burst_coverage
    monkeypatch.setattr(areas, "burst_coverage", _cover(), raising=False)
    monkeypatch.setattr(areas, "subswaths", lambda *a, **k: [2])
    from insar_pipeline import map_builder as mb, runs
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(mb, "UPLOADS", tmp_path / "uploads")
    return root


AOI = areas.box(38.30, -87.70, 38.34, -87.64, "site")


def _wp(name, route="miaplpy", **kw):
    kw.setdefault("report_areas", [("Full Satellite Imaging", "full"), ("Site", AOI)])
    return areas.write_project(name, AOI, route, "2020-01-01", "2024-12-31", track=19, direction="ascending",
                               disp_frame="04861", **kw)


def _ring_box(w, s, e, n):
    return [[w, s], [e, s], [e, n], [w, n], [w, s]]


def _holed_geojson():
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": "panel"},
            "geometry": {"type": "Polygon", "coordinates": [_ring_box(-87.69, 38.31, -87.65, 38.33),
                                                            _ring_box(-87.68, 38.315, -87.66, 38.325)]}}]}


def _cells(path, bbox, step=0.0005):
    import numpy as np
    from insar_pipeline.stack_backend import _points_in_polygons
    w, s, e, n = bbox
    lon, lat = np.meshgrid(np.arange(w, e, step), np.arange(s, n, step))
    return int(_points_in_polygons(path, lon, lat, lon.shape).sum())


# ---------------------------------------------------------------- disp-3
def test_disp_screening_gets_its_own_project_and_never_overwrites(builder_env):
    official = _wp("Site D")
    before = official.read_bytes()
    disp = _wp("Site D", "disp")
    assert official != disp and disp.stem == "site_d_disp_screening"
    assert official.read_bytes() == before                                   # official config untouched
    from insar_pipeline.config import load_config
    c_off, c_disp = load_config(str(official)), load_config(str(disp))
    assert c_off.project_dir != c_disp.project_dir and c_disp.project_dir.name == "site_d_disp_screening"
    assert "screening" in c_disp.project_title.lower() and "screening" not in c_off.project_title.lower()
    # the same name and route again: refused, nothing written
    with pytest.raises(FileExistsError, match="already exists"):
        _wp("Site D")
    assert official.read_bytes() == before
    # replacing is explicit and keeps a backup
    _wp("Site D", overwrite=True)
    assert areas.last_backup(official) is not None and areas.last_backup(official).read_bytes() == before


def test_existing_results_without_a_config_are_refused(builder_env):
    # a project run from a config kept elsewhere (pipeline/configs) has no yaml here
    geo = builder_env / "site_a_pl" / "mintpy" / "geo"
    geo.mkdir(parents=True)
    (geo / "geo_velocity.h5").write_bytes(b"")
    with pytest.raises(FileExistsError, match="MintPy results"):
        _wp("Site A PL")
    assert not (builder_env / "configs" / "site_a_pl.yaml").exists()
    assert not (builder_env / "site_a_pl" / "context").exists()           # nothing written before refusing


def test_failed_overwrite_restores_the_working_config(builder_env, monkeypatch):
    out = _wp("Gibson")
    before = out.read_bytes()
    from insar_pipeline import config as cfgmod
    monkeypatch.setattr(cfgmod, "load_config", lambda p: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ValueError):
        _wp("Gibson", overwrite=True)
    assert out.read_bytes() == before


def test_builder_exists_check_is_per_route(builder_env):
    from insar_pipeline import map_builder as mb
    _wp("Site D")
    assert mb.api_exists({"name": "Site D", "route": "miaplpy"})["exists"]
    d = mb.api_exists({"name": "Site D", "route": "disp"})
    assert d["exists"] is None and d["folder"].endswith("site_d_disp_screening")


# ---------------------------------------------------------------- gap-geolocation-6
def test_polygon_holes_survive_the_builder_round_trip(tmp_path):
    src = tmp_path / "panel.geojson"
    src.write_text(json.dumps(_holed_geojson()))
    a = areas.load_polygon(src)
    out = tmp_path / "out.geojson"
    out.write_text(json.dumps(a.geojson()))
    bbox = (-87.70, 38.30, -87.64, 38.34)
    assert _cells(out, bbox) == _cells(src, bbox) and _cells(out, bbox) < _cells(_poly(tmp_path / "filled.geojson",
                                                                                       -87.69, 38.31, -87.65, 38.33), bbox)
    # KML innerBoundaryIs, WKT holes, clip_to and merge keep them too
    kml = ("<kml><Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>-87.69,38.31 -87.65,38.31 -87.65,38.33 "
           "-87.69,38.33 -87.69,38.31</coordinates></LinearRing></outerBoundaryIs><innerBoundaryIs><LinearRing>"
           "<coordinates>-87.68,38.315 -87.66,38.315 -87.66,38.325 -87.68,38.325 -87.68,38.315</coordinates>"
           "</LinearRing></innerBoundaryIs></Polygon></Placemark></kml>")
    (tmp_path / "p.kml").write_text(kml)
    assert [len(h) for h in areas.load_polygon(tmp_path / "p.kml").holes] == [1]
    w = areas.parse_coordinates("POLYGON((-87.69 38.31,-87.65 38.31,-87.65 38.33,-87.69 38.33,-87.69 38.31),"
                                "(-87.68 38.315,-87.66 38.315,-87.66 38.325,-87.68 38.325,-87.68 38.315))")
    assert [len(h) for h in w.holes] == [1]
    m = areas.merge([areas.clip_to(a, bbox), areas.box(38.0, -87.0, 38.1, -86.9)], "all")
    assert [len(h) for h in m.holes] == [1, 0] and len(m.geojson()["features"][0]["geometry"]["coordinates"]) == 2


def test_real_mine_layer_rasterises_identically_after_the_builder(tmp_path):
    src = PKG_DIR / "context" / "coal_mine_active_gibson.geojson"       # 9 holes
    if not src.exists():
        pytest.skip("mine layer not shipped")
    a = areas.load_polygon(src)
    assert sum(len(h) for h in a.holes) >= 9
    out = tmp_path / "out.geojson"
    out.write_text(json.dumps(a.geojson()))
    assert _cells(out, a.bbox, 0.0004) == _cells(src, a.bbox, 0.0004)


def test_write_project_keeps_mine_holes_out_of_the_mine_mask(builder_env):
    mine = areas._from_geojson(_holed_geojson(), "pit")
    out = _wp("Holes", surface=mine)
    from insar_pipeline.config import load_config
    cfg = load_config(str(out))
    for f in (cfg.surface_mines_geojson, cfg.deramp_exclude_geojson):
        rings = json.loads(Path(f).read_text())["features"][0]["geometry"]["coordinates"]
        assert len(rings) == 2


def test_projected_geojson_is_refused_not_silently_dropped(tmp_path):
    utm = {"type": "Polygon", "coordinates": [[[441000, 4240000], [445000, 4240000], [445000, 4243000],
                                               [441000, 4240000]]]}
    (tmp_path / "utm.geojson").write_text(json.dumps(utm))
    with pytest.raises(ValueError, match="not longitude/latitude"):
        areas.load_polygon(tmp_path / "utm.geojson")
    kml = ("<kml><Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>441000,4240000 445000,4240000 "
           "445000,4243000 441000,4240000</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></kml>")
    (tmp_path / "utm.kml").write_text(kml)
    with pytest.raises(ValueError, match="not longitude/latitude"):
        areas.load_polygon(tmp_path / "utm.kml")


def test_geojson_that_names_its_projection_is_reprojected(tmp_path):
    pytest.importorskip("geopandas")
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 26916, always_xy=True)
    ring = [list(tr.transform(x, y)) for x, y in _ring_box(-87.69, 38.31, -87.65, 38.33)]
    doc = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::26916"}},
           "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    (tmp_path / "sp.geojson").write_text(json.dumps(doc))
    w, s, e, n = areas.load_polygon(tmp_path / "sp.geojson").bbox
    assert (w, s, e, n) == pytest.approx((-87.69, 38.31, -87.65, 38.33), abs=1e-6)


# ---------------------------------------------------------------- generality-11
def test_swaths_are_looked_up_for_the_chosen_track_on_the_padded_box(builder_env, monkeypatch):
    calls = []
    monkeypatch.setattr(areas, "burst_coverage", _cover(calls, swaths=(2, 3)), raising=False)
    out = _wp("Pad Check", pad_km=5.0)
    from insar_pipeline.config import load_config
    cfg = load_config(str(out))
    assert cfg.stack_swaths == "2 3" and calls and calls[0][1:] == (19, "ascending")
    assert calls[0][0] == pytest.approx(AOI.padded(5.0).bbox)                 # the processing box, not the area
    assert tuple(cfg.bbox) == pytest.approx(tuple(round(v, 5) for v in AOI.padded(5.0).bbox))


@pytest.mark.parametrize("fake, match", [
    (_cover(fail=ConnectionError("ASF down")), "not guessed"),
    (_cover(swaths=()), "no Sentinel-1 bursts"),
    (_cover(covered=0.62), "62.0% of the processing box"),
])
def test_no_silent_iw2_fallback(builder_env, monkeypatch, fake, match):
    monkeypatch.setattr(areas, "burst_coverage", fake, raising=False)
    monkeypatch.setattr(areas, "subswaths", lambda *a, **k: [2])
    with pytest.raises((RuntimeError, ValueError), match=match):
        _wp("No Guess")
    assert not (builder_env / "configs" / "no_guess.yaml").exists()


def test_builder_create_refuses_when_the_burst_lookup_fails(builder_env, monkeypatch):
    from insar_pipeline import map_builder as mb
    monkeypatch.setattr(areas, "burst_coverage", _cover(fail=TimeoutError("timed out")), raising=False)
    monkeypatch.setattr(areas, "subswaths", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timed out")))
    body = {"name": "Lookup Fails", "route": "miaplpy", "track": 19, "direction": "ASCENDING",
            "layers": [{"role": "area", "name": "Lookup Fails", "geojson": AOI.geojson()}]}
    with pytest.raises(RuntimeError, match="not guessed"):
        mb.api_create(body)
    assert not list((builder_env / "configs").glob("*.yaml")) if (builder_env / "configs").exists() else True


def test_burst_coverage_measures_the_union_of_distinct_bursts(monkeypatch):
    import sys
    import types
    pytest.importorskip("shapely")

    class R:
        def __init__(self, sw, bid, w, e, day):
            self.properties = {"burst": {"subswath": f"IW{sw}", "fullBurstID": bid}, "startTime": day}
            self.geometry = {"type": "Polygon", "coordinates": [_ring_box(w, 38.0, e, 39.0)]}
    hits = [R(2, "019_1_IW2", -87.5, -87.4, d) for d in ("2024-01-01", "2024-01-13")] + \
        [R(3, "019_1_IW3", -87.4, -87.35, "2024-01-01")]
    fake = types.SimpleNamespace(search=lambda **k: hits, DATASET=types.SimpleNamespace(SLC_BURST="burst"))
    monkeypatch.setitem(sys.modules, "asf_search", fake)
    box = areas.box(38.2, -87.5, 38.8, -87.3)                     # 0.2 deg wide, bursts span 0.15
    c = areas.burst_coverage(box, 19, "ascending", {"west": areas.box(38.3, -87.49, 38.4, -87.45)})
    assert c["swaths"] == [2, 3] and c["bursts"] == 2
    assert c["covered"] == pytest.approx(0.75) and c["targets"]["west"] == pytest.approx(1.0)


def test_archive_check_lists_swaths_and_coverage_per_track(builder_env, monkeypatch):
    from insar_pipeline import map_builder as mb
    monkeypatch.setattr(areas, "archive_check", lambda a, s, e, box=None: {"tracks": [
        {"direction": "ASCENDING", "track": 19, "dates": 300, "by_year": {}},
        {"direction": "DESCENDING", "track": 165, "dates": 280, "by_year": {}},
        {"direction": "DESCENDING", "track": 92, "dates": 12, "by_year": {}}], "disp_frames": {}})
    monkeypatch.setattr(areas, "gnss_near", lambda a, r=None: [])
    calls = []
    monkeypatch.setattr(areas, "burst_coverage", _cover(calls, swaths=(1, 2)), raising=False)
    r = mb.api_check({"area": {"role": "area", "geojson": AOI.geojson()}, "pad_km": 5})
    assert [t.get("swaths") for t in r["tracks"]] == [[1, 2], [1, 2], None]       # the 12-date track is skipped
    assert sorted(c[1] for c in calls) == [19, 165] and r["gnss_radius_km"] == C.GNSS_TIE_RADIUS_KM


# ---------------------------------------------------------------- gap-licensing-6
def test_nominatim_is_told_the_firm_and_a_contact(tmp_path, monkeypatch):
    monkeypatch.setenv("INSAR_SETTINGS", str(tmp_path / "s.json"))
    settings.save(firm="Example Engineering", contact="gis@example.com")
    ua = areas.user_agent()
    assert "Example Engineering" in ua and "gis@example.com" in ua and "research" not in ua
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"[]"

    def fake_urlopen(req, timeout=None):
        seen["ua"] = req.get_header("User-agent")
        return Resp()
    monkeypatch.setattr(areas.urllib.request, "urlopen", fake_urlopen)
    areas.geocode("Oakland City")
    assert seen["ua"] == ua


def test_builder_search_goes_through_the_server_with_a_warning():
    from insar_pipeline import map_builder as mb
    assert "/api/geocode" in mb.API
    assert "nominatim.openstreetmap.org" not in HTML                          # no direct browser request
    assert "not a client or site name" in HTML


# ---------------------------------------------------------------- gap-licensing-7
def test_builder_basemaps_are_credited_and_default_to_usgs():
    import re
    found = re.findall(r"(\w+): L\.tileLayer\('([^']+)',\s*\{([^}]*)\}", HTML)
    assert len(found) >= 4 and len(found) == HTML.count("L.tileLayer(")
    credits = {}
    for k, url, attrs in found:
        m = re.search(r"attribution:\s*'([^']*)'", attrs)
        assert m and m.group(1).strip(), f"{k} has no credit line"
        credits[k] = m.group(1)
    assert re.search(r"let base = bases\.usgs\.addTo\(map\)", HTML)
    assert "basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly" in HTML
    assert re.search(r'<option value="usgs" selected>', HTML)
    assert "Vantor" in credits["imagery"] and "Maxar" not in credits["imagery"]   # Esri's current credit
    assert "USGS" in credits["usgs"]


# ---------------------------------------------------------------- generality-5
@pytest.mark.parametrize("names, match", [
    ([("Pit 3: north", areas.box(38.31, -87.69, 38.32, -87.68))], "contains ':'"),
    ([("Pit 3", areas.box(38.31, -87.69, 38.32, -87.68)),
      ("Pit-3", areas.Area("t", [[(-87.66, 38.31), (-87.65, 38.31), (-87.66, 38.32), (-87.66, 38.31)]], "polygon"))],
     "differ only in punctuation"),
    ([("Gibson", AOI), ("gibson", areas.box(38.31, -87.69, 38.32, -87.68))], "two report areas"),
])
def test_report_names_that_break_or_collide_are_refused(builder_env, names, match):
    with pytest.raises(ValueError, match=match):
        _wp("Collide", report_areas=[("Full Satellite Imaging", "full")] + names)
    assert not (builder_env / "configs" / "collide.yaml").exists()


def test_report_polygons_never_share_a_reserved_file(builder_env):
    tri = areas.Area("t", [[(-87.66, 38.31), (-87.65, 38.31), (-87.66, 38.32), (-87.66, 38.31)]], "polygon")
    mine = areas._from_geojson(_holed_geojson(), "pit")
    out = _wp("Reserved", deforming=mine,
                 report_areas=[("Full Satellite Imaging", "full"), ("Deforming Area", tri)])
    from insar_pipeline.config import load_config, split_area_spec
    cfg = load_config(str(out))
    path = Path(split_area_spec(cfg.report_areas[1])[2])
    assert path.name == "report_deforming_area.geojson" and path != Path(cfg.deramp_exclude_geojson)
    assert areas.load_polygon(path).bbox == pytest.approx(tri.bbox)            # its own polygon, not the mines'


# ---------------------------------------------------------------- generality-15, usability-8, clientout-13, methodology-16
@pytest.mark.parametrize("name, stems, bad", [
    ("project_zone", (), True), ("SiteD_permit_v3", (), True), ("area-a1", (), True),
    ("permit.shp", (), True), ("Area 2", (), True), ("Project area", (), True), ("Boundary", ("Boundary",), True),
    ("Area A1", ("Area A1",), False), ("Area A2", (), False), ("Full Satellite Imaging", (), False),
    ("Pit 3", (), False),
])
def test_client_name_rules(name, stems, bad):
    assert (areas.client_name_problem(name, stems) is not None) is bad


def test_client_name_rules_cover_what_the_release_gate_blocks():
    from insar_pipeline import release
    for n in ("project_zone", "area_a1.geojson", "pit-3", "site_d_permit", "a_b"):
        if release._looks_like_file_stem(n):
            assert areas.client_name_problem(n) is not None, n


def test_uploaded_area_no_longer_takes_the_file_name(builder_env):
    import base64
    from insar_pipeline import map_builder as mb
    data = base64.b64encode(json.dumps(AOI.geojson()).encode()).decode()
    up = mb.api_outline({"files": [{"name": "project_zone.geojson", "data": data, "lastModified": 1.6e12}]})
    base = {"name": "Site B", "route": "miaplpy", "track": 19, "direction": "ASCENDING"}
    with pytest.raises(ValueError, match="file name"):                   # the stem, as the old builder sent it
        mb.api_create({**base, "layers": [{"role": "area", "name": up["label"], "upload_id": up["upload_id"]}]})
    r = mb.api_create({**base, "layers": [{"role": "area", "name": "", "upload_id": up["upload_id"]}]})
    assert r["report_areas"] == ["Full Satellite Imaging", "Site B"]
    with pytest.raises(ValueError, match="name in client report"):
        mb.api_create({**base, "name": "Site B Two", "layers": [
            {"role": "area", "name": "", "upload_id": up["upload_id"]},
            {"role": "report", "name": "Area 1", "geojson": areas.box(38.31, -87.69, 38.32, -87.68).geojson()}]})
    # the page labels the field and checks it before Create
    assert "Name in client report" in HTML and "function nameProblem" in HTML


# Site B's report_areas as they stood until the 7 Oct 2026 rename (D16):
# a fixture, so the test does not depend on the live project config.
_SITE_B_OLD_REPORT_AREAS = """\
report_areas:
- Full Satellite Imaging:full
- project_zone:/home/kylew/insar_projects/site_b/context/project_zone.geojson
"""


def test_site_b_style_config_entry_is_what_the_builder_now_refuses():
    raw = yaml.safe_load(_SITE_B_OLD_REPORT_AREAS)
    names = [C.split_area_spec(s)[0] for s in raw.get("report_areas") or []]
    assert names == ["Full Satellite Imaging", "project_zone"]
    assert areas.client_name_problem("project_zone")                        # the file-stem name is refused
    assert not areas.client_name_problem("Full Satellite Imaging")
    # the plain name it was given instead passes
    assert not areas.client_name_problem("Site B project area")


# ---------------------------------------------------------------- generality-9
def test_written_paths_follow_the_projects_root_when_it_moves(builder_env, tmp_path, monkeypatch):
    mine = areas._from_geojson(_holed_geojson(), "pit")
    tri = areas.Area("t", [[(-87.66, 38.31), (-87.65, 38.31), (-87.66, 38.32), (-87.66, 38.31)]], "polygon")
    out = _wp("Moving Site", underground=mine,
                 report_areas=[("Full Satellite Imaging", "full"), ("Site", AOI), ("North Field", tri)])
    text = out.read_text()
    assert str(builder_env) not in text and "$INSAR_PROJECTS/moving_site" in text
    moved = tmp_path / "elsewhere"
    shutil.move(str(builder_env), str(moved))
    monkeypatch.setenv("INSAR_PROJECTS", str(moved))
    cfg = load_config(str(moved / "configs" / "moving_site.yaml"))
    assert cfg.project_dir == moved / "moving_site"
    for p in (cfg.underground_mines_geojson, cfg.deramp_exclude_geojson, C.split_area_spec(cfg.report_areas[2])[2]):
        assert Path(p).is_relative_to(moved) and Path(p).exists()
    # and a configured polygon that has gone missing stops the load instead of an empty mine mask
    Path(cfg.underground_mines_geojson).unlink()
    with pytest.raises(ConfigError, match="underground_mines_geojson"):
        load_config(str(moved / "configs" / "moving_site.yaml"))


# ---------------------------------------------------------------- pools, GNSS radius, mine sources (L4a requests)
def test_new_projects_get_pools_sized_for_the_machine(builder_env, monkeypatch):
    monkeypatch.setattr(settings, "machine_resources", lambda: {"mem_gb": 7.8, "swap_gb": 2, "cores": 8,
                                                                "mem_available_gb": 6})
    out = _wp("Laptop Site")
    raw = yaml.safe_load(out.read_text())
    assert raw["mintpy_num_workers"] == 5 and raw["miaplpy_num_processors"] == 4
    assert "mintpy_max_memory_gb" not in raw and "stack_num_proc" not in raw     # left to the machine that runs it
    assert raw["gnss_tie_radius_km"] == C.GNSS_TIE_RADIUS_KM == 40.0
    assert "8 GB RAM / 8 cores" in out.read_text()


def test_mine_layer_sources_are_recorded(builder_env):
    import base64
    from insar_pipeline import map_builder as mb
    mine = areas.box(38.31, -87.69, 38.32, -87.68).geojson()
    data = base64.b64encode(json.dumps(mine).encode()).decode()
    up = mb.api_outline({"files": [{"name": "dnr_mines.geojson", "data": data, "lastModified": 1680000000000}]})
    r = mb.api_create({"name": "Sources Site", "route": "miaplpy", "track": 19, "direction": "ASCENDING", "layers": [
        {"role": "area", "name": "", "geojson": AOI.geojson()},
        {"role": "mine", "kind": "underground", "name": "dnr", "upload_id": up["upload_id"]},
        {"role": "mine", "kind": "surface", "name": "pits", "source": "Indiana DNR surface mines, 2023",
         "geojson": areas.box(38.315, -87.67, 38.32, -87.66).geojson()}]})
    cfg = load_config(r["config"])
    assert list(cfg.mine_layer_sources) == ["Underground mine workings: dnr_mines.geojson, file dated 2023-03-28",
                                            "Surface mines: Indiana DNR surface mines, 2023"]


def test_dashboard_style_area_row_named_after_a_file_takes_the_project_name(builder_env):
    # the dashboard passes (area.label, area); for an upload the label is the file's stem
    up = areas.Area("project_zone", AOI.rings, "polygon")
    out = areas.write_project("Site B", up, "miaplpy", "2020-01-01", "2024-12-31", track=19, direction="ascending",
                              report_areas=[("Full Satellite Imaging", "full"), (up.label, up)])
    cfg = load_config(str(out))
    assert [C.split_area_spec(r)[0] for r in cfg.report_areas] == ["Full Satellite Imaging", "Site B"]
    assert "'project_zone' reads like a file name" in out.read_text()
    # any OTHER report area named like a file is refused, not renamed
    with pytest.raises(ValueError, match="reads like a file name"):
        areas.write_project("Other", AOI, "miaplpy", "2020-01-01", "2024-12-31", track=19, direction="ascending",
                            report_areas=[("Full Satellite Imaging", "full"), ("north_pit", up)])
