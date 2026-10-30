"""The SAFE cleanup must protect the reference acquisition, not a date
that merely appears in the reference xml (the orbit file's production date)."""
from insar_pipeline.stack_backend import reference_dates


def test_reference_date_is_the_safe_not_the_orbit_file(tmp_path):
    ref = tmp_path / "reference"
    (ref / "IW2").mkdir(parents=True)
    (ref / "IW2.xml").write_text(
        "<property name='orbitfile'><value>/o/S1A_OPER_AUX_POEORB_OPOD_20210310T052706_V20160105T225943_20160107T005943.EOF"
        "</value></property><property name='safe'><value>['/s/S1A_IW_SLC__1SSV_20160105T234741_20160105T234746_009366_00D8CF_BC1E.SAFE']"
        "</value></property>")
    (ref / "IW2" / "burst_01.slc.vrt").write_text(
        '<SourceFilename relativeToVRT="1">../../../slc/S1A_IW_SLC__1SSV_20160105T234741_20160105T234746_009366_00D8CF_BC1E.SAFE/'
        "measurement/s1a-iw2-slc-vv-20160105t234741-001.tiff</SourceFilename>")
    assert reference_dates(ref) == {"20160105"}


def _stack(tmp_path, dates):
    import types
    stack = tmp_path / "stack"
    ref = stack / "reference" / "IW2"; ref.mkdir(parents=True)
    (ref / "burst_01.slc.vrt").write_text(
        f"../../../slc/S1A_IW_SLC__1SSV_{dates[0]}T234741_{dates[0]}T234746_009366_00D8CF_BC1E.SAFE/x.tiff")
    for d in dates:
        m = stack / "merged" / "SLC" / d; m.mkdir(parents=True)
        (m / f"{d}.slc.full").write_bytes(b"\0" * 16)
        (m / f"{d}.slc.full.xml").write_text(
            '<property name="width"><value>25001</value></property><property name="length"><value>4190</value></property>')
    (stack / "crop_window.json").write_text('{"r0": 10, "r1": 20, "c0": 30, "c1": 40, "length": 1396, "width": 2272}')
    return types.SimpleNamespace(stack_dir=stack, stack_reference_date="", stack_azimuth_looks=3, stack_range_looks=11,
                                 project_dir=tmp_path, miaplpy_work_dir="", miaplpy_dir=tmp_path / "miaplpy")


def test_blank_reference_date_is_inferred_and_kept(tmp_path, monkeypatch):
    """Site B: with stack_reference_date blank, the merged-SLC cleanup deleted
    every raster, then aligned_subset found none. Now the reference is inferred
    and kept, and the crop comes from the xml sidecars alone."""
    import h5py
    import numpy as np
    from insar_pipeline import miaplpy_backend as mb
    dates = ["20160105", "20160117", "20160129"]
    cfg = _stack(tmp_path, dates)
    assert mb.stack_reference(cfg) == "20160105"
    work = tmp_path / "work" / "inputs"; work.mkdir(parents=True)
    with h5py.File(work / "slcStack.h5", "w") as f:
        f["date"] = np.array([d.encode() for d in dates])
        f["slc"] = np.ones((3, 300, 300), np.complex64)
    monkeypatch.setattr(mb, "work_dir", lambda c: tmp_path / "work")
    mb.delete_merged_slcs_after_load(cfg)
    left = sorted(p.parent.name for p in (cfg.stack_dir / "merged" / "SLC").glob("*/*.slc.full"))
    assert left == ["20160105"]
    for p in (cfg.stack_dir / "merged" / "SLC").glob("*/*.slc.full"):
        p.unlink()                                   # even with no raster left at all
    assert mb.aligned_subset(cfg) == (30, 60, 330, 440)


def test_recorded_reference_choice_is_reused_by_a_new_process(tmp_path):
    """Site B: no reference point in the YAML. choose_reference picks one and
    records it; every later process (MintPy onward, products, results) must use
    that pixel, not let MintPy choose its own."""
    import json
    import yaml
    from insar_pipeline.config import load_config
    # a phase-linked project with no reference point (the example config's
    # settings until the example configs left the repository, 28 Oct 2026)
    src = {"backend": "miaplpy", "project_dir": str(tmp_path / "p"), "reference_lalo": "",
           "bbox": [-95.89174, 29.65493, -95.60826, 29.86507], "start_date": "2021-10-01",
           "end_date": "2026-09-30", "track": 34, "orbit_direction": "ascending", "mintpy_deramp": "no",
           "stack_virtual_merge": "False"}
    c = tmp_path / "c.yaml"; c.write_text(yaml.safe_dump(src))
    assert load_config(str(c)).reference_lalo == ""
    (tmp_path / "p" / "stack").mkdir(parents=True)
    (tmp_path / "p" / "stack" / "reference_choice.json").write_text(json.dumps(
        {"configured": None, "chosen": {"lat": 38.330376493, "lon": -87.351410683}}))
    assert load_config(str(c)).reference_lalo == "38.330376,-87.351411"
    # after a restart the kept reference is recorded as "configured"; it must still be found
    (tmp_path / "p" / "stack" / "reference_choice.json").write_text(json.dumps(
        {"configured": {"lat": 38.330376, "lon": -87.351411}, "kept_configured": True}))
    assert load_config(str(c)).reference_lalo == "38.330376,-87.351411"
