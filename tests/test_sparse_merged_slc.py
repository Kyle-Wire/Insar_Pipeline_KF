"""Merged SLCs kept to the phase-linking window (11 Nov 2026, Kyle: "we
definitely want that cropping feature"): miaplpy_backend.sparsify_merged_slc
rewrites a date's <date>.slc.full as a sparse file of the same name, size and
header holding only the window. These guard that the bytes inside the window
are bit for bit the original (NaN included), that everything outside reads
zero, that the disk really holds less, that a window left by another area or
an older geometry is never used, that a load reaching outside a kept window
is refused, and that the planner credits the saving."""
import json
import os

import numpy as np
import pytest

from insar_pipeline import miaplpy_backend as mb
from insar_pipeline.config import PipelineConfig

L, W = 512, 1024            # full-resolution merged frame
LOOKS = 2                   # 2 x 2 looks: the crop window is in the 256 x 512 multilooked grid
D = "20230104"


def _xml(path, length, width, dtype="CFLOAT"):
    path.write_text(f'<imageFile><property name="width"><value>{width}</value></property>'
                    f'<property name="length"><value>{length}</value></property>'
                    f'<property name="data_type"><value>{dtype}</value></property></imageFile>')


def _project(tmp_path, win=(40, 140, 100, 300)):
    cfg = PipelineConfig(project_dir=tmp_path / "proj", bbox=[-115.36, 36.06, -115.11, 36.26], start_date="2023-01-01",
                         end_date="2023-08-31", backend="miaplpy")
    cfg.stack_azimuth_looks = cfg.stack_range_looks = LOOKS
    s = cfg.stack_dir
    geom = s / "merged" / "geom_reference"
    geom.mkdir(parents=True)
    for n in ("lat", "lon", "hgt"):
        (geom / f"{n}.rdr").write_bytes(b"\0" * 8)
    r0, r1, c0, c1 = win
    (s / "crop_window.json").write_text(json.dumps({
        "r0": r0, "r1": r1, "c0": c0, "c1": c1, "length": L // LOOKS, "width": W // LOOKS,
        "bbox_wsen": list(cfg.bbox), "margin_deg": cfg.stack_bbox_margin_deg,
        "geometry_mtime": os.path.getmtime(geom / "lat.rdr")}))
    return cfg


def _merged(cfg, date=D, seed=0):
    d = cfg.stack_dir / "merged" / "SLC" / date
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((L, W)) + 1j * rng.standard_normal((L, W))).astype("<c8")
    a[50, 250] = complex(np.nan, 1.0)                             # a NaN inside the window survives as it was
    full = d / f"{date}.slc.full"
    a.tofile(full)
    _xml(d / f"{date}.slc.full.xml", L, W)
    return full, a


def test_inside_the_window_is_the_original_and_outside_is_zero(tmp_path):
    cfg = _project(tmp_path)
    full, a = _merged(cfg)
    y0, y1, x0, x1 = mb.sparse_window(cfg)
    assert (y0, y1, x0, x1) == (80, 280, 200, 600)                # the multilooked window x the looks
    before = full.stat().st_blocks
    gb = mb.sparsify_merged_slc(cfg, D)
    b = np.fromfile(full, dtype="<c8").reshape(L, W)
    assert full.stat().st_size == L * W * 8                       # same size: every size check still passes
    assert np.array_equal(b[y0:y1, x0:x1].view("<u8"), a[y0:y1, x0:x1].view("<u8"))   # bit for bit, NaN too
    out = np.ones((L, W), bool)
    out[y0:y1, x0:x1] = False
    assert not b[out].any()
    if full.stat().st_blocks < before:                            # a filesystem with holes (ext4, tmpfs)
        assert gb > 0 and full.stat().st_blocks * 512 < 0.5 * L * W * 8
    m = json.loads((full.parent / f"{D}{mb.SPARSE_MARKER}").read_text())
    assert m["window_yx"] == [y0, y1, x0, x1]
    assert mb._merged_slc_verified(cfg, D)
    assert mb.sparsify_merged_slc(cfg, D) == 0.0                  # once only
    assert not list(full.parent.glob("*.sparse_tmp"))


def test_a_window_from_another_area_or_older_geometry_is_not_used(tmp_path):
    cfg = _project(tmp_path)
    full, a = _merged(cfg)
    cfg.bbox = [-115.40, 36.06, -115.11, 36.26]                   # the area changed since the window was written
    assert mb.sparse_window(cfg) is None and mb.sparsify_merged_slc(cfg, D) == 0.0
    cfg = _project(tmp_path / "b")
    full, a = _merged(cfg)
    lat = cfg.stack_dir / "merged" / "geom_reference" / "lat.rdr"
    os.utime(lat, (1, 1))                                         # the geometry was merged again
    assert mb.sparse_window(cfg) is None
    cfg.miaplpy_sparse_merged_slc = False
    assert mb.sparse_window(cfg) is None


def test_a_load_reaching_outside_a_kept_window_is_refused(tmp_path):
    cfg = _project(tmp_path)
    _merged(cfg)
    mb.sparsify_merged_slc(cfg, D)
    mb.check_sparse_windows(cfg)                                  # the same window: fine
    cw = cfg.stack_dir / "crop_window.json"
    rec = json.loads(cw.read_text())
    rec["c1"] += 10                                               # a wider area reads past what was kept
    cw.write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="merge those dates again|Merge those dates again"):
        mb.check_sparse_windows(cfg)


def test_non_complex_rasters_and_unverified_files_are_left_alone(tmp_path):
    cfg = _project(tmp_path)
    full, _ = _merged(cfg)
    _xml(full.with_name(full.name + ".xml"), L, W, dtype="FLOAT")
    assert mb.sparsify_merged_slc(cfg, D) == 0.0
    cfg2 = _project(tmp_path / "b")
    full2, _ = _merged(cfg2)
    with open(full2, "r+b") as f:                                  # short: not a verified merge
        f.truncate(100)
    assert mb.sparsify_merged_slc(cfg2, D) == 0.0


def test_the_geometry_hook_waits_for_lat_lon_and_hgt(tmp_path, monkeypatch):
    """The window is written only after the lat, lon and hgt merges have
    finished in this execution (a full-size file may still be being written)."""
    from insar_pipeline import stack_backend as sb
    cfg = _project(tmp_path)
    (cfg.stack_dir / "crop_window.json").unlink()
    calls = []
    monkeypatch.setattr(mb, "ensure_crop_window", lambda c: calls.append(1) or True)
    hook = sb._merge_cleanup_hook(cfg, "run_12_merge_reference_secondary_slc")
    for g in ("lat", "lon"):
        hook(f"SentinelWrapper.py -c /x/configs/config_merge_{g}", {"returncode": 0})
    assert not calls
    hook("SentinelWrapper.py -c /x/configs/config_merge_hgt", {"returncode": 0})
    assert calls == [1]


def test_the_planner_credits_the_sparse_merged_slcs(tmp_path):
    from insar_pipeline import stack_backend as sb
    src = open(sb.__file__).read()
    assert 'getattr(cfg, "miaplpy_sparse_merged_slc", True)' in src and "plan.n_dates * f" in src


def test_storage_counts_what_a_sparse_file_holds(tmp_path):
    """insar storage and the Files page count a sparse merged SLC at what it
    holds on disk, not its length (Explorer shows the length)."""
    from insar_pipeline import file_manager, storage
    f = tmp_path / "proj" / "big.slc.full"
    f.parent.mkdir()
    with open(f, "wb") as h:
        h.truncate(200 * 1024 * 1024)                              # 200 MB long, nothing written
        h.seek(0)
        h.write(b"x" * 8192)
    small = tmp_path / "proj" / "note.txt"
    small.write_text("abc")
    if f.stat().st_blocks * 512 >= f.stat().st_size:
        pytest.skip("this filesystem keeps no holes")
    assert storage.du(f) < 1024 * 1024
    assert storage.du(small) == 3                                  # ordinary files: their apparent bytes, as before
    assert storage.du(tmp_path / "proj") < 1024 * 1024
    assert file_manager._sizes(tmp_path)["proj"] < 1024 * 1024
