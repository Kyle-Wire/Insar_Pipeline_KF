"""7 Oct 2026 final verification, group kmz (client_kmz.py): decisions M3
(the reissue note), M4 (feature figures: the outline caveat and the rate's
range), E9 in the KMZ (rates to 0.01 in/yr as the page, every total with its
95% range, feature totals to 0.1 in), the point verdict by the per-cell rule
of rate_signal_to_noise.tif, the whole area's signal / threshold words, the
GNSS stations' uncertainty and M2's 500 m 'next to' rule in standout_text.
Each test here fails on the code in backups/pre_final_20261007/.
"""
import html as htmlmod
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from insar_pipeline import client_kmz as ck                       # noqa: E402

THREE_DP_IN = re.compile(r"\d\.\d{3} in")


def _plain(s):
    return re.sub(r"\s+", " ", htmlmod.unescape(re.sub(r"<[^>]+>", " ", s)))


REP = {"anchored": True, "span_years": 10.56, "control_ground_label": "farmland", "dates": ["20160105", "20260730"],
       "n_dates": 270, "title": "Site A", "orbit": "ASCENDING", "track": 92, "look_angle_deg": 39.5,
       "los_east_over_up": 0.8, "mine_layers": ["mine areas"], "mine_layer_info": [], "route": "miaplpy",
       "resolution": {"posting_m": 15.0, "support_m": [92.3, 239.1],
                      "text": "Points are 15 m apart; each value combines similar ground within about 90 m "
                              "east-west and 240 m north-south."},
       "gnss": {"background": {"vertical_mm_yr": -0.76, "vertical_uncertainty_mm_yr": 1.01}}}

# Site A's final figures (trust_check_20261007/site_a_pl client_report.json)
FULL = {"name": "Full Satellite Imaging", "kind": "full", "cells": 3157158, "area_km2": 712.6,
        "footprint_km2": 734.91, "measured_percent": 97.0, "regional_gnss_vertical_mm_yr": -0.76,
        "regional_gnss_uncertainty_mm_yr": 1.01, "vertical_mm_yr": -1.59, "total_mm": -16.8,
        "total_at_average_rate_mm": -16.8, "total_range95_mm": [-41.4, 7.8], "range95_mm_yr": [-3.92, 0.74],
        "signal_to_noise": 0.68, "reading": "No measurable movement of the region as a whole",
        "recent_rate_mm_yr": 0.2, "recent_range95_mm_yr": [-2.81, 3.21], "recent_window": ["20240802", "20260730"]}

FEAT = {"sign": -1, "area_km2": 23.71, "lat": 38.2405, "lon": -87.6473, "vertical_relative_mm_yr": -16.3,
        "total_relative_mm": -172.0, "total_at_average_rate_mm": -172.0, "total_range95_mm": [-243.0, -104.0],
        "farmland_spread_1sigma_mm_yr": 3.19, "recent_rate_mm_yr": -10.5, "recent_range95_mm_yr": [-19.94, 0.09],
        "recent_window": ["20240802", "20260730"], "mines": {"key": "none"},
        "text": "It moves about 0.64 in/yr (16 mm/yr) down relative to the surrounding farmland."}


# ------------------------------------------------------------------ E9 precision and ranges
def test_area_balloon_rates_to_two_decimals_like_the_page():
    """'Vertical, per year -0.063 in', 'Regional ground motion -0.030 in/yr',
    'Last two years +0.008 in/yr': 3 decimals beside a range of +/- 0.09 in/yr."""
    t = _plain(ck.area_description(FULL, REP))
    assert not THREE_DP_IN.search(t), THREE_DP_IN.findall(t)
    assert "-0.06 in (-1.6 mm)" in t                              # the rate, as the page prints it
    assert "-0.03 in/yr (-0.8 ± 1.0 mm/yr, 1 sigma)" in t         # the regional GNSS rate
    assert "Last two years (2024–2026) +0.01 in/yr (+0.2 mm/yr), 95% -0.11 to +0.13 in/yr" in t
    assert "-0.66 in (-17 mm), 95% -1.63 to +0.31 in" in t        # the area total, to 0.01 in with its range


def test_whole_area_signal_to_threshold_is_its_own_range_test():
    """The whole area is not tested against farmland: its number is its
    rate (satellite plus GNSS) over its 95% uncertainty, not '(1 or more
    stands out)'."""
    t = _plain(ck.area_description(FULL, REP))
    assert "0.7 (its rate, satellite plus GNSS, ÷ its 95% uncertainty; 1 or more means its range excludes zero)" in t
    assert "1 or more stands out" not in t
    area = {"name": "Area A2", "kind": "area", "testable": True, "vertical_mm_yr": -6.4, "total_mm": -67.6,
            "range95_mm_yr": [-14.0, 1.2], "signal_to_noise": 0.4, "reading": "Within normal ground variation",
            "cells": 1000, "area_km2": 2.0}
    assert "0.4 (1 or more stands out)" in _plain(ck.area_description(area, REP))


def test_feature_balloon_total_to_a_tenth_with_range_and_rate_range():
    """M4 / E9: '-10.55 in' sat beside 'moved 10.6 in' in the same balloon,
    and the feature's rate was the one figure with no range."""
    t = _plain(ck.feature_description(1, FEAT, REP))
    assert not THREE_DP_IN.search(t), THREE_DP_IN.findall(t)
    assert "-6.8 in (-172 mm), 95% -9.6 to -4.1 in" in t
    # the rate's range: -16.3 +/- 1.96 x 3.19 mm/yr (the spread of ordinary farmland of its size)
    assert "95% range -0.89 to -0.40 in/yr (-22.6 to -10.0 mm/yr), from the spread of ordinary farmland" in t
    assert "-0.64 in (-16.3 mm)" in t
    assert "Last two years (2024–2026) -0.41 in/yr (-10.5 mm/yr)" in t
    # a feature from a report with no total range: no bare total (E9)
    old = {k: v for k, v in FEAT.items() if k != "total_range95_mm"}
    assert "Measured over the record" not in _plain(ck.feature_description(1, old, REP))


def test_feature_balloon_says_the_outline_reads_stronger_once():
    """M4: the outline is drawn from the same data, so its figures read
    somewhat stronger than the ground moved - said once (not again when the
    report's own paragraph already says it)."""
    t = _plain(ck.feature_description(1, FEAT, REP))
    assert "outline was drawn from these same measurements" in t and "somewhat stronger" in t
    said = dict(FEAT, text=FEAT["text"] + " Its outline was drawn from the same data, so these figures read "
                                          "somewhat stronger than the ground truly moved.")
    t = _plain(ck.feature_description(1, said, REP))
    assert t.count("stronger") == 1


def test_feature_rate_range_prefers_the_reports_own():
    assert ck.feature_rate_range(dict(FEAT, range95_mm_yr=[-21.0, -11.0])) == [-21.0, -11.0]
    # the report's own key (client_report: rate_range95_mm_yr, on the
    # footprint test's frame) - the range the page's paragraph prints
    assert ck.feature_rate_range(dict(FEAT, rate_range95_mm_yr=[-23.0, -9.8])) == [-23.0, -9.8]
    lo, hi = ck.feature_rate_range(FEAT)
    assert lo == pytest.approx(-16.3 - 1.96 * 3.19) and hi == pytest.approx(-16.3 + 1.96 * 3.19)
    assert ck.feature_rate_range({"vertical_relative_mm_yr": -5.0}) is None


def test_point_total_carries_its_range_and_rates_two_decimals():
    """'+2.26 in (+57 mm)' had no range beside a rate range of -0.46 to
    +0.89 in/yr; the point's rate printed to 0.001 in."""
    pv = ck.point_values(5.46, 6.22, 4.0, 1.01, True)
    t = _plain(ck.point_description("Area A2", 38.3, -87.5, pv, 10.4, 0.8, None, resolution=REP["resolution"]))
    assert not THREE_DP_IN.search(t), THREE_DP_IN.findall(t)
    lo, hi = pv["range"]
    assert (f"At the average rate over the record (10.4 yr) +2.2 in (+57 mm), 95% {lo * 10.4 / 25.4:+.1f} to "
            f"{hi * 10.4 / 25.4:+.1f} in") in t
    assert "+0.21 in (+5.5 mm)" in t
    # the verdict to 0.01: a value just under 1 never reads '1.0 (within ...)'
    pv = ck.point_values(-2.97, -2.97, 3.0 / 1.96, None, False)
    t = _plain(ck.point_description("Field", 38.3, -87.5, pv, 10.4, 0.8, None))
    assert "Signal ÷ threshold 0.99 (within normal variation" in t


def test_values_beside_a_limit_never_print_at_the_limit():
    """'0.70 (below 0.7: fails the second quality check)' for 0.699, and
    '1.00 (within normal variation)' for 0.996 (Site A / Site B previews)."""
    pv = ck.point_values(-3.0, -2.988, 3.0 / 1.96, None, False)          # signal / threshold 0.996
    t = _plain(ck.point_description("Field", 38.3, -87.5, pv, 10.4, 0.8, None, unwrap=0.699, gate=0.7))
    assert "Signal ÷ threshold 0.99 (within normal variation" in t
    assert "Unwrapping consistency 0.69 (below 0.7: fails" in t
    assert ck._beside_limit(0.99, 1.0) == "0.99" and ck._beside_limit(1.004, 1.0) == "1.00"
    # only a value that would round up to the limit is rounded down
    # (review of 7 Oct 2026: every value below it was, 0.68 read '0.67' and
    # a whole area's 0.68 read '0.6' beside the page's '0.7')
    assert ck._beside_limit(0.678, 1.0) == "0.68" and ck._beside_limit(0.68, 1.0, 1) == "0.7"
    assert ck._beside_limit(0.456, 0.7) == "0.46" and ck._beside_limit(0.99999999999, 1.0) == "0.99"
    assert ck._beside_limit(0.97, 1.0, 1) == "0.9" and ck._beside_limit(0.6996, 0.7) == "0.69"


def test_the_average_rate_total_carries_its_range_too():
    """E9: '-15.4 in (-390 mm) at the average rate' stood bare beside the
    measured total's range (Site B Feature 2)."""
    f = dict(FEAT, vertical_relative_mm_yr=-37.3, farmland_spread_1sigma_mm_yr=4.24, total_relative_mm=-329.0,
             total_at_average_rate_mm=-390.0, total_range95_mm=[-436.0, -259.0])
    t = _plain(ck.feature_description(2, f, dict(REP, span_years=10.45)))
    lo, hi = ((-37.3 - 1.96 * 4.24) * 10.45 / 25.4, (-37.3 + 1.96 * 4.24) * 10.45 / 25.4)
    assert f"-15.4 in (-390 mm) at the average rate (95% {lo:+.1f} to {hi:+.1f} in)" in t
    big = {"name": "A", "kind": "area", "testable": True, "vertical_mm_yr": -11.5, "total_mm": -100.0,
           "total_at_average_rate_mm": -120.0, "range95_mm_yr": [-18.0, -5.0], "total_range95_mm": [-190.0, -10.0],
           "signal_to_noise": 1.2, "reading": "Movement detected", "cells": 100, "area_km2": 1.0}
    t = _plain(ck.area_description(big, dict(REP, span_years=10.4)))
    assert "-4.72 in (-120 mm) at the average rate (95% -7.37 to -2.05 in)" in t


def test_point_with_no_rate_is_not_tested():
    pv = ck.point_values(float("nan"), float("nan"), 3.0, None, False)
    assert pv["faded"] is True and pv["signal_to_threshold"] is None and pv["rel_range"] is None
    t = _plain(ck.point_description("Field", 38.3, -87.5, pv, 10.4, 0.8, None))
    assert "nan" not in t.lower() and "no rate at this cell" in t


def test_chart_titles_give_the_rate_to_two_decimals_and_a_total_only_with_its_range():
    t = ck.series_title("Feature 1", -25.7, -268.0, True, "farmland", [-300.0, -241.0], nd_total=1)
    assert t.startswith("Feature 1: -1.01 in/yr (-25.7 mm/yr)")
    assert "-10.6 in measured over the record (95% -11.8 to -9.5 in)" in t and "farmland" in t
    t = ck.series_title("Whole area", -1.17, -12.0, False)
    assert "-0.05 in/yr" in t and "measured over the record" not in t          # no range: no total
    assert not THREE_DP_IN.search(ck.series_title("Area", -1.17, -12.0, False, "farmland", [-30.0, 6.0]))


# ------------------------------------------------------------------ M3 the reissue note
REVISION = {"number": 1, "supersedes": "2026-09-28",
            "changes": ["Detection limits are now at 80% power (Area A2 0.59 -> 0.72 in/yr).",
                        "Feature 1 is listed as worth a look."]}


def test_document_balloon_and_name_say_revision_and_what_changed(tmp_path):
    cfg = types.SimpleNamespace(project_dir=tmp_path / "site_a_pl", backend="miaplpy", report_revision=REVISION)
    d = _plain(ck.doc_description(cfg, REP, False, [], False))
    assert "Revision 1, replacing the report issued 28 Sep 2026." in d
    for c in REVISION["changes"]:
        assert c in d
    assert ck.doc_name(cfg, REP, False, False) == "Site A: ground movement 2016–2026, Revision 1 (DRAFT)"
    # read from the report record when the config carries none
    cfg2 = types.SimpleNamespace(project_dir=tmp_path / "site_a_pl", backend="miaplpy")
    d2 = _plain(ck.doc_description(cfg2, dict(REP, report_revision=REVISION), True, [], False))
    assert "Revision 1, replacing the report issued 28 Sep 2026." in d2
    # the config's own reader words it (one heading for the page, the PDF and the KMZ)
    from insar_pipeline import config as cf
    if hasattr(cf, "report_revision"):
        assert ck.revision_heading(ck.revision_record(cfg, REP)) == cf.report_revision(REVISION)["heading"]
    # a first issue says nothing of revisions
    assert "Revision" not in _plain(ck.doc_description(cfg2, REP, True, [], False))
    assert "Revision" not in ck.doc_name(cfg2, REP, True, False)


def test_revision_dates_are_written_plainly():
    import datetime as dt
    assert ck._date_text("20260928") == "28 Sep 2026" and ck._date_text(dt.date(2026, 9, 28)) == "28 Sep 2026"
    assert ck.revision_heading({"number": 2, "supersedes": "7 Oct 2026"}) == \
        "Revision 2, replacing the report issued 7 Oct 2026"
    assert ck.revision_record(types.SimpleNamespace(report_revision=None), {}) is None


# ------------------------------------------------------------------ M2 'next to' only within 500 m
def test_standout_text_says_next_to_only_within_500_m():
    so = {"lat": 38.2401, "lon": -87.6462, "scale_km": 4.0, "vertical_relative_mm_yr": -14.3,
          "inside_mapped_panels": False, "within_panel_buffer": True, "buffer_m": 2000.0}
    t = ck.standout_text(so, 1, ["mine areas"])
    assert "next to" not in t and "within 2.0 km of a mapped mine area" in t
    assert "next to mapped mine areas" in ck.standout_text(dict(so, buffer_m=500.0), 1, ["mine areas"])


# ------------------------------------------------------------------ end to end: the per-cell rule
@pytest.fixture(scope="module")
def kmz_project(tmp_path_factory):
    """test_fix_L9b's small project, its client report, and a context whose
    relative field alternates cell by cell (a checkerboard of +/- 1.5 x the
    cell null), so a 3 x 3 mean and its centre cell differ as they do in
    windows that mix coherence classes."""
    import test_fix_L9b as L9b
    from insar_pipeline import client_report as cr
    from insar_pipeline import results as rs
    from insar_pipeline.panel_history import _Context
    root = tmp_path_factory.mktemp("lastkmz")
    cfg = L9b.make_project(root)
    cr.build(cfg)
    D = cfg.project_dir / "deliverables"
    rep = json.loads((D / "client_report.json").read_text())
    # the GNSS stations' placemarks (the test tie has none): one with its sigma
    rep.setdefault("gnss", {})["stations"] = [{"site": "INVI", "lat": 38.3, "lon": -87.55, "distance_km": 12.0,
                                               "vertical_over_record_mm_yr": {"rate_mm_yr": -1.09, "sigma_mm_yr": 0.91},
                                               "used_for_background": True}]
    ctx = _Context(cfg)
    rs.report_farmland(ctx, rep)
    model = json.loads((D / "uncertainty_model.json").read_text())
    null = cr.cell_sigma_map(rep.get("cell_null_by_class"), ctx.kcls, model)
    rr, cc = np.indices(ctx.r_cm.shape)
    ctx.r_cm = ctx.r_cm + np.where((rr + cc) % 2 == 0, 1.5, -1.5) * np.nan_to_num(null)
    out = root / "raster"
    out.mkdir()
    snr_path = cr.write_signal_to_noise(cfg, rep, ctx, null, out)
    mp = pytest.MonkeyPatch()
    mp.setattr(ck, "PER_AREA", 30)
    mp.setattr(ck, "PER_FEATURE", 10)
    try:
        kmz = ck.build(cfg, out_dir=root / "preview", rep=rep, ctx=ctx)
    finally:
        mp.undo()
    import rasterio
    with rasterio.open(snr_path) as r:
        snr = r.read(1)
    kml = zipfile.ZipFile(kmz).read("doc.kml").decode()
    return types.SimpleNamespace(cfg=cfg, rep=rep, ctx=ctx, kml=kml, snr=snr, L9b=L9b)


def _points(kml):
    """(style, balloon, lon, lat) as Google Earth shows them (13 Oct 2026: one
    shared balloon per style, each point's values in ExtendedData)."""
    return ck.expand_placemarks(kml)


def test_point_verdict_is_the_signal_to_noise_rasters_at_that_cell(kmz_project):
    """Every point's signal / threshold is rate_signal_to_noise.tif / 1.96
    at its own cell, and 'stands out' / the faded dot follow the raster's
    1.96 (the 3 x 3 mean over the centre's null disagreed on ~3% of points)."""
    p, L9b = kmz_project, kmz_project.L9b
    n = n_out = n_in = 0
    for style, desc, lon, lat in _points(p.kml):
        txt = _plain(desc)
        if "Not measurable" in txt:
            continue
        r = int(round((float(lat) - L9b.Y0) / L9b.DY - 0.5))
        c = int(round((float(lon) - L9b.X0) / L9b.DX - 0.5))
        s_rast = float(p.snr[r, c])
        got = re.search(r"Signal ÷ threshold\s+(\d+\.(\d+))", txt)
        s2t = float(got.group(1))
        # printed to 0.01, below 1 rounded down (0.1 before 7 Oct)
        tol = 0.011 if len(got.group(2)) >= 2 else 0.051
        assert s2t == pytest.approx(s_rast / 1.96, abs=tol), (r, c, txt)
        if abs(s_rast - 1.96) > 0.02:
            assert ("(stands out)" in txt) == (s_rast >= 1.96)
            assert style.endswith("f") == (s_rast < 1.96)
            n_out += s_rast >= 1.96
            n_in += s_rast < 1.96
        n += 1
    assert n > 20 and n_out > 3 and n_in > 3


def test_built_kmz_has_no_three_decimal_inches_and_stations_carry_sigma(kmz_project):
    kml = _plain(kmz_project.kml)
    assert not THREE_DP_IN.search(kml), THREE_DP_IN.findall(kml)[:5]
    assert "-1.1 ± 0.9 mm/yr (1 sigma) over the record's dates" in htmlmod.unescape(kmz_project.kml)
    # every point total carries its range
    for _, desc, _, _ in _points(kmz_project.kml):
        t = _plain(desc)
        if "At the average rate over the record" in t:
            assert re.search(r"At the average rate over the record \([\d.]+ yr\) [-+]?\d+\.\d in \([-+]?\d+ mm\), "
                             r"95% [-+]?\d+\.\d to [-+]?\d+\.\d in", t), t


# ------------------------------------------------------------------ review of 7 Oct 2026
def test_feature_balloon_range_is_the_one_the_page_prints():
    """A rebuilt report carries the feature's rate range (rate_range95_mm_yr,
    on the footprint test's frame: Site A Feature 1's total range over
    the years gives -23.0 to -9.8 mm/yr); the page's paragraph prints it, so
    the balloon's table must too, not 1.96 x the spread (-22.6 to -10.0)."""
    from insar_pipeline import report_insights as ri
    f = dict(FEAT, rate_range95_mm_yr=[-23.0, -9.8])
    f["text"] = ri.rate_range_words(f["vertical_relative_mm_yr"], f["rate_range95_mm_yr"])
    t = _plain(ck.feature_description(1, f, REP))
    assert "95% range -0.91 to -0.39 in/yr (-23.0 to -9.8 mm/yr)" in t
    assert "-22.6 to -10.0" not in t
    assert "95% range 0.39 to 0.91 in/yr" in t                     # the paragraph's, the same range
    # the average-rate total's range is that range times the years
    f2 = dict(f, total_at_average_rate_mm=-200.0)
    t = _plain(ck.feature_description(1, f2, REP))
    assert (f"at the average rate (95% {-23.0 * 10.56 / 25.4:+.1f} to {-9.8 * 10.56 / 25.4:+.1f} in)") in t


def test_feature_outline_caveat_is_the_reports_own_sentence():
    """One wording on the page, the PDF and the KMZ (report_insights.OUTLINE_CAVEAT)."""
    from insar_pipeline import report_insights as ri
    if not hasattr(ri, "OUTLINE_CAVEAT"):
        pytest.skip("report_insights has no OUTLINE_CAVEAT")
    t = _plain(ck.feature_description(1, FEAT, REP))
    assert ri.OUTLINE_CAVEAT in t and t.count("stronger") == 1
    # a rebuilt paragraph already ends with it: said once
    said = dict(FEAT, text=FEAT["text"] + " " + ri.OUTLINE_CAVEAT)
    assert _plain(ck.feature_description(1, said, REP)).count("stronger") == 1


def test_area_signal_to_threshold_just_under_one_never_reads_one():
    """0.96 printed '1.0 (1 or more stands out)' beside 'Within normal
    ground variation' (the area's detected is signal / threshold >= 1)."""
    area = {"name": "Area A2", "kind": "area", "testable": True, "vertical_mm_yr": -6.4, "total_mm": -67.6,
            "range95_mm_yr": [-14.0, 1.2], "signal_to_noise": 0.96, "reading": "Within normal ground variation",
            "cells": 1000, "area_km2": 2.0}
    t = _plain(ck.area_description(area, REP))
    assert "0.9 (1 or more stands out)" in t and "1.0 (1 or more" not in t
    assert "1.2 (1 or more stands out)" in _plain(ck.area_description(dict(area, signal_to_noise=1.24), REP))
    full = dict(FULL, signal_to_noise=0.97)
    assert "0.9 (its rate, satellite plus GNSS" in _plain(ck.area_description(full, REP))
