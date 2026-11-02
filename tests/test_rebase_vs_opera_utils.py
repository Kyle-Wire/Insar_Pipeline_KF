"""
Cross-check this pipeline's reference-epoch stitching against OPERA's own.

§8 of the PRD asked for `opera-utils` to be evaluated, because its
`rebase_timeseries` is the same step that caused the §4.1 regression.
This is that evaluation, as a test rather than a one-off.

WHAT IT ESTABLISHES

1. With no missing data the two implementations agree EXACTLY. That is
   the independent confirmation that the stitching arithmetic here is
   right - two codebases, written from the same specification, producing
   bit-identical output.

2. With missing data they deliberately differ, and this is the
   interesting part. OPERA offers two NaN policies and this pipeline
   implements a third:

     propagate  (OPERA default) - a NaN in the crossover frame makes the
                pixel NaN for the rest of the record. This IS the §4.1
                bug. OPERA can live with it because they do not mask
                connected components; the NaNs they see are rare product
                nodata. Add per-epoch conncomp masking - 58% of the grid
                per granule over Site A farmland - and the same code
                destroys the scene.

     omit       - the missing offset is treated as ZERO, i.e. "no ground
                movement happened during that segment". The pixel
                survives but its absolute displacement is wrong by the
                entire accumulated offset, which on this site is tens of
                millimetres. It converts a visible failure into an
                invisible one.

     last valid (this pipeline) - carry the pixel's most recent VALID
                absolute displacement. The error is whatever the ground
                did between that epoch and the boundary, typically days,
                so millimetres at most.

   The tests below assert each of those three behaviours, so if anyone
   changes the policy the consequence is spelled out rather than
   discovered later in a velocity map.

The comparisons against OPERA's code are SKIPPED (pytest.skip, so `pytest
-rs` lists them) when opera-utils is not installed - it is not in the insar
environment today. Until 5 Oct 2026 they printed "(skipped ...)" and
returned, which pytest counted as passes: "5 passed" with three of them
never run. "Ours" is the PRODUCTION assemble_stack fed these granules
through monkeypatched reads, not a copy of its carry (the copy differed: it
kept the previous carry where a pixel had never been valid; production
carries the NaN).

Run: python -m pytest -rs tests/test_rebase_vs_opera_utils.py
"""
import contextlib
import importlib.util
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import disp_backend as db       # noqa: E402

_REBASE_PATH = (Path(sys.prefix) / "lib" / f"python{sys.version_info.major}."
                f"{sys.version_info.minor}" / "site-packages" / "opera_utils"
                / "disp" / "_rebase.py")


def _load_opera_rebase():
    """
    Import OPERA's rebase without importing the whole package.

    `opera_utils.disp.__init__` pulls in zarr, which is not needed here
    and which would mean adding a dependency to an environment that has
    already been broken once by a careless install (see PRD section 9).
    """
    if not _REBASE_PATH.exists():
        return None
    pkg = types.ModuleType("ou_disp")
    pkg.__path__ = []
    sys.modules.setdefault("ou_disp", pkg)
    utils = types.ModuleType("ou_disp._utils")
    utils._clamp_chunk_dict = lambda *a, **k: None
    sys.modules.setdefault("ou_disp._utils", utils)
    spec = importlib.util.spec_from_file_location("ou_disp._rebase", _REBASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ou_disp._rebase"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001
        return None
    return mod


OPERA = _load_opera_rebase()

N_EPOCHS = 30
SEG_LEN = 6
NY = NX = 8


def _scenario(seed=0, nan_at=None):
    """
    Granule values as OPERA ships them: relative to each own reference.

    The reference date of each segment is the LAST SECONDARY DATE of the
    segment before it. That is not an assumption - it was verified across
    all 16 boundaries of the real Site A frame, where `last_sec ==
    next_ref` held exactly every time. Both implementations rely on it,
    so a synthetic scenario that gets it wrong tests the wrong thing.
    """
    rng = np.random.default_rng(seed)
    base = datetime(2020, 1, 1)
    sec_dates = [base + timedelta(days=12 * i) for i in range(N_EPOCHS)]
    seg = np.arange(N_EPOCHS) // SEG_LEN

    truth = np.stack([
        np.full((NY, NX), -0.012 * (d - base).days / 365.25) for d in sec_dates
    ]).astype("float32")
    truth += rng.normal(0, 0.001, truth.shape).astype("float32")

    # Segment 0 is referenced to its own first epoch; segment s>0 to the
    # last epoch of segment s-1.
    ref_index = {0: 0}
    ref_index.update({s: s * SEG_LEN - 1 for s in range(1, int(seg.max()) + 1)})
    ref_dates = [sec_dates[ref_index[int(s)]] for s in seg]

    raw = np.stack([truth[i] - truth[ref_index[int(seg[i])]]
                    for i in range(N_EPOCHS)])
    if nan_at:
        for i, (r, c) in nan_at:
            raw[i, r, c] = np.nan
    return raw, ref_dates, truth


@contextlib.contextmanager
def _granule_reads(arrays: dict):
    """Serve `arrays` {fileName: {layer: array}} to assemble_stack; no network."""
    saved = {k: getattr(db, k) for k in ("_data_url", "_geometry_from_granule", "_open_remote", "_read_granule")}

    def no_network(*a, **k):
        raise OSError("network disabled in tests")
    db._data_url = lambda p: "https://example.invalid/g.nc"
    db._geometry_from_granule = lambda u: (None, None)
    db._open_remote = no_network
    db._read_granule = lambda cfg, url, fname, win, layers, attempts=4: (
        arrays[fname], (0.0, 1.0, 0.0, 1.0, 32616), None, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(db, k, v)


def _ours(raw, ref_dates):
    """This pipeline's accumulation: the production assemble_stack."""
    base = datetime(2020, 1, 1)
    sec_dates = [base + timedelta(days=12 * i) for i in range(raw.shape[0])]
    names = [f"OPERA_L3_DISP-S1_IW_F09999_VV_{r:%Y%m%d}T000000Z_{s:%Y%m%d}T000000Z_v1.0_20250101T000000Z.nc"
             for r, s in zip(ref_dates, sec_dates)]
    arrays = {nm: {"displacement": raw[i].astype("float32")} for i, nm in enumerate(names)}
    cfg = types.SimpleNamespace(project_dir=None, disp_apply_corrections=False,
                                disp_mask_connected_components=False, disp_flag_unwrap_jumps=False,
                                disp_repair_boundary_steps=False, disp_repair_boundary_datum=False,
                                disp_decimate=1)
    with _granule_reads(arrays):
        A = db.assemble_stack(cfg, [types.SimpleNamespace(properties={"fileName": nm}) for nm in names])
    return A["ts"]


def test_agrees_exactly_with_opera_utils_when_nothing_is_missing():
    """Two independent implementations, same specification, same answer."""
    if OPERA is None:
        pytest.skip("opera-utils is not importable in this environment")
    raw, refs, _ = _scenario()
    theirs = OPERA.rebase_timeseries(raw.copy(), refs)
    ours = _ours(raw.copy(), refs)
    assert np.allclose(ours, theirs, atol=1e-6, equal_nan=True), (
        f"max difference {np.nanmax(np.abs(ours - theirs)):.3e} m")


def test_both_recover_the_true_displacement_without_gaps():
    raw, refs, truth = _scenario()
    ours = _ours(raw.copy(), refs)
    # Rebasing returns displacement relative to the first epoch.
    expect = truth - truth[0]
    assert np.allclose(ours, expect, atol=1e-5)


def test_opera_default_policy_reproduces_the_section_4_1_regression():
    """
    Their default is `propagate`, and propagate is the bug.

    Not a criticism of opera-utils: they do not mask connected
    components, so the NaNs they encounter are rare. It becomes fatal the
    moment per-epoch masking is added upstream, which is exactly what
    happened here.
    """
    if OPERA is None:
        pytest.skip("opera-utils is not importable in this environment")
    # One pixel is missing in the single frame that crosses a boundary.
    raw, refs, _ = _scenario(nan_at=[(SEG_LEN - 1, (3, 3))])
    theirs = OPERA.rebase_timeseries(raw.copy(), refs)
    tail = theirs[SEG_LEN:, 3, 3]
    assert np.isnan(tail).all(), (
        "propagate should kill the pixel for the rest of the record")
    # And the rest of the scene is untouched, which is why it goes unnoticed.
    assert np.isfinite(theirs[SEG_LEN:, 0, 0]).all()


def test_our_policy_keeps_the_pixel_and_stays_close_to_the_truth():
    raw, refs, truth = _scenario(nan_at=[(SEG_LEN - 1, (3, 3))])
    ours = _ours(raw.copy(), refs)
    tail = ours[SEG_LEN:, 3, 3]
    assert np.isfinite(tail).all(), "the pixel must survive"

    expect = (truth - truth[0])[SEG_LEN:, 3, 3]
    err = np.abs(tail - expect).max()
    # The error is one epoch of ground motion, not the whole offset.
    assert err < 0.002, f"carried error {err * 1000:.1f} mm is too large"


def test_omit_policy_would_lose_the_accumulated_offset():
    """
    The failure mode `omit` trades for: the pixel survives, wrong.

    Zeroing the crossover means asserting the ground did not move during
    that segment. Over a decade of accumulating segments that is a large,
    entirely invisible error - which is why this pipeline does not use it.
    """
    if OPERA is None:
        pytest.skip("opera-utils is not importable in this environment")
    raw, refs, truth = _scenario(nan_at=[(SEG_LEN - 1, (3, 3))])
    theirs = OPERA.rebase_timeseries(raw.copy(), refs, nan_policy="omit")
    tail = theirs[SEG_LEN:, 3, 3]
    assert np.isfinite(tail).all(), "omit should keep the pixel"

    expect = (truth - truth[0])[SEG_LEN:, 3, 3]
    err = np.abs(tail - expect).max()
    ours_err = np.abs(_ours(raw.copy(), refs)[SEG_LEN:, 3, 3] - expect).max()
    assert err > ours_err, (
        f"omit ({err * 1000:.1f} mm) should be worse than last-valid "
        f"({ours_err * 1000:.1f} mm)")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-rs", "-q"]))
