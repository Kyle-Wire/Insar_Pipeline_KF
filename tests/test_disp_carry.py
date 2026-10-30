"""
The reference-epoch carry is the most fragile step in the DISP-S1 backend.

DISP-S1 resets its reference date roughly every six months, so each
segment has to be offset by the displacement reached at the boundary. That
offset is an ARRAY, and NaN + x = NaN - so if the carry is built from a
frame that has already been masked, every pixel masked in that one granule
is lost for the rest of the record. Over a decade that is 16 boundaries
compounding, and a per-granule mask of a few percent becomes near-total
loss.

This is not hypothetical. It is what reduced a real Site A run to 0.8%
coverage, and because the surviving pixels were a single non-subsiding
corner of the AOI it also inverted the apparent sign of the result and
sent the investigation after a sign-convention bug that did not exist.

The test asserts the property that matters and is cheap to check on a real
run too: per-epoch valid fraction should be roughly FLAT across the stack.
A monotonic decay towards zero means the carry is leaking.

The fix is tested on the PRODUCTION assemble_stack, fed synthetic granules
through monkeypatched reads (until 5 Oct 2026 these tests exercised a copy
of the carry, so a regression in assemble_stack would have passed them).
The old bug is kept as a small model (`_old_carry`) only to show what the
diagnosis was.

Run: python -m pytest tests/test_disp_carry.py
"""
import types
from datetime import datetime, timedelta

import numpy as np

from insar_pipeline import disp_backend as db

NY = NX = 40
N_EPOCHS = 254
N_BOUNDARIES = 16
TRUE_RATE_M_PER_DAY = -0.012 / 365.25      # ~ -0.47 in/yr, Site A-like
BAD_FRACTION = 0.15                        # conncomp==0 share per granule
START = datetime(2016, 7, 1)


def _synthetic_stack(seed=0):
    """Granule values as OPERA ships them: segment r > 0 is referenced to
    the LAST epoch of segment r - 1 (its anchor), segment 0 to the start."""
    rng = np.random.default_rng(seed)
    days = (np.arange(N_EPOCHS) + 1) * 12
    refs = np.repeat(np.arange(N_BOUNDARIES + 1),
                     int(np.ceil(N_EPOCHS / (N_BOUNDARIES + 1))))[:N_EPOCHS]
    bad = rng.random((N_EPOCHS, NY, NX)) < BAD_FRACTION
    truth = (TRUE_RATE_M_PER_DAY * days)[:, None, None] * np.ones((1, NY, NX))
    anchor = {r: int(np.argmax(refs == r)) - 1 for r in np.unique(refs) if r > 0}
    rel = np.stack([truth[i] - (truth[anchor[refs[i]]] if refs[i] > 0 else 0.0)
                    for i in range(N_EPOCHS)])
    return days, refs, bad, rel


def _old_carry(days, refs, bad, rel):
    """MODEL of the original bug (carry = the previous, masked frame)."""
    frames = []
    carry = None
    last_ref = None
    for i in range(N_EPOCHS):
        d = np.where(bad[i], np.nan, rel[i]).astype("float32")
        if last_ref is not None and refs[i] != last_ref:
            carry = frames[-1].copy()
        if carry is not None:
            d = d + carry
        frames.append(d)
        last_ref = refs[i]
    return np.stack(frames)


def _production(monkeypatch, days, refs, bad, rel):
    """The real assemble_stack on these granules (reads monkeypatched)."""
    sec = [START + timedelta(days=int(d)) for d in days]
    first = {r: int(np.argmax(refs == r)) for r in np.unique(refs)}
    refdate = {r: (sec[first[r] - 1] if r > 0 else START) for r in first}
    names = [f"OPERA_L3_DISP-S1_IW_F09999_VV_{refdate[refs[i]]:%Y%m%d}T234758Z_"
             f"{sec[i]:%Y%m%d}T234758Z_v1.0_20250101T000000Z.nc" for i in range(N_EPOCHS)]
    arrays = {nm: {"displacement": np.where(bad[i], np.nan, rel[i]).astype("float32")}
              for i, nm in enumerate(names)}

    def no_network(*a, **k):
        raise OSError("network disabled in tests")
    monkeypatch.setattr(db, "_data_url", lambda p: "https://example.invalid/g.nc")
    monkeypatch.setattr(db, "_geometry_from_granule", lambda u: (None, None))
    monkeypatch.setattr(db, "_open_remote", no_network)
    monkeypatch.setattr(db, "_read_granule", lambda cfg, url, fname, win, layers, attempts=4:
                        (arrays[fname], (0.0, 1.0, 0.0, 1.0, 32616), None, None))
    cfg = types.SimpleNamespace(project_dir=None, disp_apply_corrections=False,
                                disp_mask_connected_components=False, disp_flag_unwrap_jumps=False,
                                disp_repair_boundary_steps=False, disp_repair_boundary_datum=False,
                                disp_decimate=1)
    results = [types.SimpleNamespace(properties={"fileName": nm}) for nm in names]
    return db.assemble_stack(cfg, results)


def _per_epoch_valid(ts):
    return np.isfinite(ts).reshape(ts.shape[0], -1).mean(axis=1)


def test_carrying_the_previous_frame_ratchets_coverage_away():
    """Guard the diagnosis: the old behaviour really does collapse."""
    ts = _old_carry(*_synthetic_stack())
    pe = _per_epoch_valid(ts)
    assert pe[:5].mean() > 0.8, "start should be ~1 - BAD_FRACTION"
    assert pe[-5:].mean() < 0.2, "the bug should destroy late coverage"


def test_carrying_the_last_valid_value_keeps_coverage_flat(monkeypatch):
    """The fix, in production: a pixel only dies when it is bad everywhere."""
    A = _production(monkeypatch, *_synthetic_stack())
    pe = _per_epoch_valid(A["ts"])
    assert pe[:5].mean() > 0.8
    assert pe[-5:].mean() > 0.8
    # No systematic decay across the decade, and the ratchet check agrees.
    assert abs(pe[-5:].mean() - pe[:5].mean()) < 0.05
    assert not A["coverage_ratchet"]["fired"]
    assert len(A["anchor_idx"]) == N_BOUNDARIES


def test_fix_recovers_the_true_rate_over_the_whole_grid(monkeypatch):
    days, refs, bad, rel = _synthetic_stack()
    ts = _production(monkeypatch, days, refs, bad, rel)["ts"]
    t = days / 365.25
    flat = ts.reshape(N_EPOCHS, -1)
    finite = np.isfinite(flat)
    usable = finite.sum(axis=0) >= 0.5 * N_EPOCHS

    # Every pixel should survive, not just a corner of the grid.
    assert usable.all(), f"only {usable.sum()} of {usable.size} pixels usable"

    vel = np.full(flat.shape[1], np.nan)
    for j in np.where(usable)[0]:
        m = finite[:, j]
        A = np.vstack([t[m], np.ones(m.sum())]).T
        vel[j] = np.linalg.lstsq(A, flat[m, j], rcond=None)[0][0]

    truth = TRUE_RATE_M_PER_DAY * 365.25
    assert np.isfinite(vel).all()
    # Sign is the thing that was doubted; assert it explicitly.
    assert np.nanmedian(vel) < 0, "subsidence must read negative"
    assert abs(np.nanmedian(vel) - truth) < 0.15 * abs(truth)
