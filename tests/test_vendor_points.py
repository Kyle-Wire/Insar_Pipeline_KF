"""Vendor point-export comparison (21 Oct 2026): the export's bookkeeping, and our
series put through the same bookkeeping."""
import numpy as np

from insar_pipeline import vendor_points as vp

DATES = ["20241001", "20241013", "20241025", "20241106", "20241118", "20241130"]


def _write(tmp_path, rows):
    head = list(vp.META_COLUMNS) + DATES
    lines = [",".join(f'"{h}"' for h in head)]
    for pid, vals in rows:
        disp = [v for v in vals if v is not None][-1]
        steps = vp.column_steps_days(DATES)
        yrs = sum(s for s, v in zip(steps, vals) if v is not None) / vp.YEAR_DAYS
        meta = [f'"{pid}"', "-87.4", "38.3", "-87.4", "38.3", f"{disp / yrs:.4f}", '"Line of Sight"', f"{disp}"]
        lines.append(",".join(meta + ["" if v is None else str(v) for v in vals]))
    p = tmp_path / "cumulative_displacement_Line_of_Sight.csv"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_rate_is_displacement_over_accepted_time(tmp_path):
    p = _write(tmp_path, [("1", [None, -0.1, None, -0.3, -0.35, None]),
                          ("2", [None, 0.0, 0.1, 0.1, 0.2, 0.25])])
    ex = vp.read_export(p)
    assert ex.dates == DATES and ex.values.shape == (2, 6)
    assert np.allclose(vp.accepted_years(ex), [36 / 365.25, 60 / 365.25])
    rid = vp.rate_identity(ex)
    assert rid["match_0p0015"] == 2


def test_blank_intervals_add_nothing_on_both_sides(tmp_path):
    # the vendor accepted 1013, 1106, 1118 for point 1 (blank 1025: carried flat)
    p = _write(tmp_path, [("1", [None, -0.1, None, -0.3, -0.35, None])])
    ex = vp.read_export(p)
    ours = np.array([[0.0, -0.2, -1.0, -1.1, -1.2, -5.0]])       # big drop in the rejected 1013->1025
    g = vp.gated(ex, ours, DATES)
    # accepted intervals: 1001->1013, 1106->1118 (1025->1106 holds a blank only at 1025, which is
    # inside 1013->1025; 1025->1106 is accepted at 1106)
    acc = g["accepted"][0]
    assert acc.tolist() == [True, False, True, True, False]
    assert np.isclose(g["their_d"][0], -0.1 + (-0.3 - -0.1) + (-0.35 - -0.3))
    assert np.isclose(g["our_d"][0], -0.2 + (-1.1 - -1.0) + (-1.2 - -1.1))   # the -0.8 drop is not counted
    assert np.isclose(g["years"][0], (12 + 12 + 12) / 365.25)


def test_a_date_we_lack_is_merged_into_the_spanning_interval(tmp_path):
    p = _write(tmp_path, [("1", [None, -0.1, -0.2, -0.3, -0.4, -0.5]),
                          ("2", [None, -0.1, None, -0.3, -0.4, -0.5])])
    ex = vp.read_export(p)
    ours_dates = [d for d in DATES if d != "20241025"]
    ours = np.tile(np.array([0.0, -0.1, -0.3, -0.4, -0.5]), (2, 1))
    g = vp.gated(ex, ours, ours_dates)
    assert len(g["intervals"]) == 4 and g["intervals"][1] == (1, 3)
    assert g["accepted"][0, 1] and not g["accepted"][1, 1]      # point 2 rejected 1025, inside the merged interval
    assert np.isclose(g["interval_days"][1], 24)
    s = vp.summarize(ex, g)
    assert s["common"]["their_gated_in"] is not None and s["rate_identity"]["points"] == 2
