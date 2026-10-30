"""NISAR single-look images for the phase-linked route (nisar_gslc.py,
PLAN_nisar_phase_linked Phase 1, 28 Oct 2026) and the Products-only note when
a NISAR project's crops are on another grid than its config (Phase 0).
Synthetic data only; no network."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from insar_pipeline import nisar_backend as nb, nisar_gslc as ng

NAME = "NISAR_L2_PR_GSLC_032_019_A_021_4005_DHDH_A_20261001T121114_20261001T121149_P05023_N_F_J_001.h5"


# ---------------------------------------------------------------- Phase 0
def test_products_note_when_the_crops_are_on_another_grid(tmp_path):
    cfg = SimpleNamespace(project_dir=str(tmp_path), nisar_posting_m=15.0)
    assert nb.posting_note(cfg) is None                                   # no stack record yet
    (tmp_path / "nisar_gunw").mkdir()
    rec = tmp_path / "nisar_gunw" / "nisar_stack.json"
    rec.write_text(json.dumps({"written": "2026-10-26 10:57:27", "settings": {"nisar_posting_m": 80.0}}))
    note = nb.posting_note(cfg)
    assert "80 m grid" in note and "2026-10-26" in note and "Whole pipeline" in note and "15 m" in note
    rec.write_text(json.dumps({"settings": {"nisar_posting_m": 15.0}}))
    assert nb.posting_note(cfg) is None


# ---------------------------------------------------------------- names and the stack
def test_granule_name_fields():
    f = ng.parse_name(NAME)
    assert (f["track"], f["direction"], f["frame"], f["date"], f["mode"], f["pols"]) == \
        (19, "ascending", 21, "20261001", "4005", "DHDH")
    assert ng.parse_name("NISAR_L2_PR_GUNW_x.h5") is None


def _rec(track, d, frame=21, direction="ascending", counter=1, crid="P05023"):
    return {"track": track, "direction": direction, "frame": frame, "date": d, "counter": counter, "crid": crid,
            "name": f"{track}_{frame}_{d}_{counter}_{crid}"}


def test_one_track_and_frame_one_granule_per_date():
    recs = [_rec(19, "20251030"), _rec(19, "20251111"), _rec(19, "20251111", counter=2),
            _rec(19, "20251123", frame=22), _rec(91, "20260601"), _rec(27, "20260601", direction="descending")]
    st = ng.choose_stack(recs, track=19, direction="ascending")
    assert [r["date"] for r in st] == ["20251030", "20251111"]
    assert st[1]["counter"] == 2                                           # the reprocessed granule wins
    assert ng.choose_stack(recs, track=99) == []


def test_window_covers_the_box_plus_margin():
    x = 500000.0 + 5.0 * np.arange(4000)                                   # UTM 13N, 5 m
    y = 4310000.0 - 5.0 * np.arange(4000)
    bbox = (-104.89, 38.88, -104.88, 38.89)
    r0, r1, c0, c1 = ng.window_for(x, y, 32613, bbox, margin_m=500)
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 32613, always_xy=True)
    xs, ys = tr.transform([bbox[0], bbox[2]], [bbox[1], bbox[3]])
    assert x[c0] <= min(xs) - 495 and x[c1 - 1] >= max(xs) + 495
    assert y[r1 - 1] <= min(ys) - 495 and y[r0] >= max(ys) + 495
    with pytest.raises(ValueError, match="not on this GSLC"):
        ng.window_for(x, y, 32613, (-100.0, 30.0, -99.9, 30.1))


def _crop(path, rec, bbox, x, y, pol="HH", margin=500.0):
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["slc"] = np.ones((y.size, x.size), np.complex64)
        f["mask"] = np.ones((y.size, x.size), np.uint8)
        f["x"], f["y"] = x, y
        f.attrs.update({"granule": rec["name"], "crop_version": ng.CROP_VERSION, "polarization": pol,
                        "margin_m": margin, "bbox": np.asarray(bbox, float), "epsg": 32613})


def test_a_crop_is_reused_only_when_everything_matches(tmp_path):
    rec, bbox = _rec(19, "20251030"), (-104.89, 38.88, -104.88, 38.89)
    x, y = np.arange(10.0), -np.arange(8.0)
    p = tmp_path / "c.h5"
    _crop(p, rec, bbox, x, y)
    assert ng.reusable(p, rec, bbox, 500.0, "HH")
    assert not ng.reusable(p, rec, bbox, 500.0, "HV")
    assert not ng.reusable(p, rec, bbox, 1000.0, "HH")
    assert not ng.reusable(p, rec, (-104.9, 38.88, -104.88, 38.89), 500.0, "HH")
    assert not ng.reusable(p, _rec(19, "20251030", counter=2), bbox, 500.0, "HH")


def test_same_grid_names_a_crop_on_another_grid(tmp_path):
    rec, bbox = _rec(19, "20251030"), (0, 0, 1, 1)
    x, y = np.arange(10.0), -np.arange(8.0)
    a, b, c = (tmp_path / f"{k}.h5" for k in "abc")
    _crop(a, rec, bbox, x, y)
    _crop(b, rec, bbox, x, y)
    _crop(c, rec, bbox, x + 5, y)
    g = ng.same_grid([a, b, c])
    assert g["same"] is False and g["different"] == ["c.h5"] and g["shape"] == [8, 10]
    assert ng.same_grid([a, b])["same"] is True


def test_coherence_of_a_known_pair():
    rng = np.random.default_rng(0)
    n = 200
    s = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(np.complex64)
    ramp = np.exp(1j * np.linspace(0, 2 * np.pi, n))[None, :].astype(np.complex64)
    g, ph = ng.coherence(s * ramp, s, k=5)
    assert np.median(g) > 0.99                                          # the same scene, a smooth phase
    assert abs(np.median(ph[:, 100]) - np.angle(ramp[0, 100])) < 0.05
    noise = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(np.complex64)
    g2, _ = ng.coherence(s, noise, k=5)
    assert np.median(g2) < 0.35                                         # independent scenes (5 x 5 bias ~0.18)


def test_valid_cells_follow_the_gslc_mask():
    mask = np.array([[0, 1, 2, 255]], np.uint8)
    slc = np.array([[1, 1, 0, 1]], np.complex64)
    assert ng.valid_cells(mask).tolist() == [[False, True, True, False]]
    assert ng.valid_cells(mask, slc).tolist() == [[False, True, False, False]]


def test_preview_skips_an_empty_date_and_pairs_across_it(tmp_path):
    import h5py
    rng = np.random.default_rng(1)
    base = (rng.normal(size=(60, 60)) + 1j * rng.normal(size=(60, 60))).astype(np.complex64)
    x, y = 500000.0 + 5 * np.arange(60), 4300000.0 - 5 * np.arange(60)
    reads = []
    for d, empty in (("20260814", False), ("20260826", False), ("20260907", True), ("20260919", False)):
        p = tmp_path / "crops" / f"{d}.h5"
        p.parent.mkdir(exist_ok=True)
        with h5py.File(p, "w") as f:
            f["slc"] = np.zeros_like(base) if empty else base
            f["mask"] = np.zeros(base.shape, np.uint8) if empty else np.ones(base.shape, np.uint8)
            f["x"], f["y"] = x, y
            f.attrs["epsg"] = 32613
        reads.append({"date": d, "file": str(p)})
    (tmp_path / "gslc_stack.json").write_text(json.dumps(
        {"reads": reads, "empty": [{"date": "20260907", "valid_share": 0.0, "cause": "mask 0"}]}))
    res = ng.preview(tmp_path, k=5)
    assert [r["pair"] for r in res["pairs"]] == ["20260814_20260826", "20260826_20260919"]
    assert res["pairs"][1]["days"] == 24 and res["pairs"][1]["median_coherence"] > 0.99
    assert res["dates_left_out"] == ["20260907"]
