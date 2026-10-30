"""
26 Oct 2026: the results viewer turned date windows off on Site E ("The
points' own series do not reproduce the delivered figures ... rate 0.054
mm/yr") and on Site E NISAR (0.13). Its whole-record check fitted each
point's own-cell series on significance.years over every date, but MintPy
fitted the delivered velocity on its own decimal years without the dates it
judged too noisy (mintpy/exclude_date.txt: 20250510 at Site E). The check
now fits the delivered way (field_points.fit_axis); a table built before
gets the axis when the viewer loads it (add_fit_axis), so it needs no rebuild.
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import field_points as fp           # noqa: E402
from insar_pipeline import significance as sg            # noqa: E402

DATES = [f"2025{m:02d}{d:02d}" for m in range(1, 13) for d in (5, 17, 28)][:30] + ["20260110", "20260122"]


def _table(tmp_path, excluded):
    """Own-cell series whose stored rate is MintPy's: its axis, without the
    excluded dates (one of them an outlier)."""
    M = tmp_path / "mintpy"
    M.mkdir(parents=True, exist_ok=True)
    if excluded:
        (M / "exclude_date.txt").write_text("\n".join(excluded) + "\n")
    rng = np.random.default_rng(3)
    tm = fp.mintpy_years(DATES) - fp.mintpy_years(DATES)[0]
    N = 50
    rate = rng.normal(0, 8, N)
    C = (rate[:, None] * tm[None, :] + rng.normal(0, 1.5, (N, len(DATES)))).astype(np.float32)
    keep = np.array([d not in excluded for d in DATES])
    C[:, ~keep] += 25.0                                           # the noisy date MintPy left out
    stored = fp.slopes(tm[keep], C[:, keep].astype(float))        # MintPy's velocity
    meta = {"dates": DATES, "t": [float(x) for x in sg.years(DATES)]}
    fd = {"cell_los": C, "los_rel": stored, "change_los": (C[:, -1] - C[:, 0]).astype(float), "meta": meta}
    return types.SimpleNamespace(mintpy_dir=M), fd


def test_the_check_fits_the_way_mintpy_fitted_the_delivered_rate(tmp_path):
    cfg, fd = _table(tmp_path, ["20250517"])
    old = fp.whole_record_check(np.asarray(fd["meta"]["t"]), fd["cell_los"].astype(float), fd["los_rel"], fd["change_los"])
    assert old[0] > 0.01                                          # the old check: windows off
    fp.add_fit_axis(cfg, fd)
    m = fd["meta"]
    assert m["fit_exclude"] == ["20250517"] and len(m["t_fit"]) == len(DATES)
    assert m["whole_record_check_mm_yr"] < 1e-4 and m["whole_record_change_check_mm"] < 1e-4
    # a table that already has the axis is left as it is
    before = dict(m)
    fp.add_fit_axis(cfg, fd)
    assert fd["meta"] == before


def test_mintpy_years_is_mintpys_axis():
    from mintpy.utils import ptime
    got = fp.mintpy_years(DATES)
    assert np.allclose(got, ptime.yyyymmdd2years(DATES), atol=1e-12)


def test_no_exclude_file_means_every_date_on_mintpys_axis(tmp_path):
    cfg, fd = _table(tmp_path, [])
    t, ex = fp.fit_axis(cfg, DATES)
    assert ex == [] and t[0] == 0 and np.all(np.isfinite(fp.fit_t({"t": [0] * len(DATES), "t_fit": list(t),
                                                                   "fit_exclude": [], "dates": DATES})))
    fp.add_fit_axis(cfg, fd)
    assert fd["meta"]["whole_record_check_mm_yr"] < 1e-4
