# Experiment A truth, digitised (PLAN phase 2, T1), 8 Oct 2026

Everything here was read from public mine reports and drawings. Nothing was
estimated by eye except the value of one grid line per drawing axis (the grid
labels are drawn as glyph outlines, not text) and which survey pegs belong to
which line. Every number can be rebuilt from the PDFs in `sources/` with the
scripts in `scripts/` (about 2 minutes; see "Rebuild" below).

## Files

| file | what it is | read by |
|---|---|---|
| `tahmoor_west_gnss_units.csv` | the 17 Tahmoor West GNSS units, one row per unit and epoch, Up/East/North in mm with sigmas | `ground_truth.load_points_csv`, `insar validate --points` |
| `tahmoor_west_gnss_daily.csv` | every value plotted in the 17 figures: each day's daily value and MSEC's 7-day average, per component | people (the full record) |
| `tahmoor_west_gnss_calibration.json` | per figure: both passes' axis calibrations, styles, the position fit, the sigmas, dropped epochs, figure notes | people, tests |
| `panels.geojson` | 19 longwall panel outlines (Appin Areas 7 and 9, Tahmoor West, North and South) with dates, depth of cover, draw distance, window status, provenance | the harness (stable ring, panel statistics), `deforming_build.py` |
| `survey_lines.geojson` | the 4 survey lines that carry protocol A's end-of-panel maxima (LW902, LW903, LW904, LW708) | the harness (`appin_peak_ratio`) |
| `georef_drawings.json` | each drawing's georeference: grid lines used and their values, fit, residuals; the affine ties | people, tests |
| `panel_checks.json` | outlines against the printed void dimensions, the same panels on other drawings, two absolute checks | people, tests |
| `sources.json` | every source PDF: URL, bytes, sha256, and a same-size HEAD check of the URL on 8 Oct 2026 | people, tests |
| `checks/` | `gnss_checks.json`, `deforming_checks.json`, `review_checks.json` (the review's independent checks: South32's MGA2020 drawing, the date bars), `qc_map.png`, and `overlay_TW_*.png` (the listed epochs drawn back on each rendered figure) | people, tests |
| `../../configs/experiments/context/appin_tahmoor_deforming.geojson` | the replacement deramp-exclusion polygon (the APPROX file is kept) | the A1, A2 and A3 configs since 8 Oct 2026, before any run (protocol A amendment `2026-10-08/a-deramp-polygon`) |

## 1. Tahmoor West GNSS units

**Source.** Tahmoor West Report 8 (1 Jan - 30 Jun 2023), its appendix MSEC1263
Rev 21 (Aug 2023), Fig. G07-R21 to Fig. GSR17S-R21, pages 89-105. These are
the 17 units the notes and protocol A count. Each figure plots the change in
easting, northing and height (+ve = subsidence) from the start of LW W1 (15 Nov
2019), or from the unit's installation, to Aug 2023: every daily value and the
7-day running average. Report 8 plots the whole history, so it is the one
source. Reports 1-4 and 6 carry no GNSS figures; Report 5 has eight older
figures (Rev 08, including Sites 14 and 23, which are not among the 17);
Report 7 (Rev 20, Nov 2022) has the same 17 and is used as a check.

**Two independent passes** (protocol A: "each plot digitised twice,
independently"):

- **A, vector.** The figures are vector drawings: every daily value is a
  vertex of a stroked path, and so is the 7-day average. The axes are
  calibrated from the vector tick marks and their numeric labels (least
  squares; residual at the ticks <= 0.1 mm and < 0.1 day). Nothing is
  resampled.
- **B, raster.** The page rendered at 600 dpi; the axes calibrated from the
  gridlines found in the image (not from the vector); the 7-day curve read
  column by column from its colour.
- A - B at the listed epochs, per unit: RMS 0.09-1.0 mm in Up (largest at
  Site 24, through the steep W4 drop) and 0.05-0.62 mm in East and North.

**Value written.** MSEC's 7-day running average from pass A (the report: "the
7-day running average readings are the most appropriate reflection of measured
changes"). It is the *centred* 7-day mean of the plotted daily values: in Up
the plotted average matches it to 0.03-1.55 mm RMS per unit, a trailing mean
only to 1.6-12.9 mm. (This tests the dates of the daily values against those
of the average. It does not test the date axis itself, because both curves
use the same axis. That is checked below against the mining bars.)

**Epochs.** Each unit's first 7-day value (2019-11-17/18 for the units that ran
from the start of W1; later for 12A, 20, 21, 24, SR17N, SR17S), then the 1st
and 15th of every month while the unit has data, up to Jul 2023. An epoch is
written only when at least 4 of the 7 days around it have a plotted daily
value; the average line is drawn straight across data gaps and those stretches
are not used (`dropped` in the calibration file). The harness takes each
unit's first epoch on or after the window start as its zero, so the window cut
is the harness's.

**Signs and frames.** `up_mm` = -(plotted height change). `east_mm` and
`north_mm` are as plotted, in MGA grid directions. The frame is plate-fixed:
units on the stable side show no common northward drift (ITRF plate motion
here is about +55 mm/yr north), so protocol A's rule for horizontal digitised
truth (CA07's change minus its own linear rate) applies. MGA grid north here
is 1.35 deg from true north (meridian convergence at 150.59 E, 34.17 S, zone
56). Rotate E/N by that angle before projecting them onto a LOS vector. For
Site 24's 130 mm of horizontal movement, that changes East by about 3 mm.

**Sigmas** (columns, per component; Up has no suffix, East and North
`_east`, `_north`). Protocol A amendment `2026-10-08/p23-1` (8 Oct 2026,
decided by Claude for Kyle, delegated, before any result) replaced the
half-gridline floor with the measured terms, per point:

- `digitising_rms_mm`: the measured A - B RMS (0.1-1.0 mm), the
  repeatability.
- `half_line_width_mm`: half the plotted line width in data units: half the
  stroke width of the 7-day-average paths read (0.504 pt in every figure,
  the legend sample's width too) times the value axis's mm per pt
  (0.15-1.33 mm).
- `calibration_mm`: the axis-calibration residual at the point: the value
  axis's RMS at its labelled ticks, in quadrature with the date axis's RMS
  (days) times the curve's slope at the epoch (mm per day): 0.00-0.51 mm.
- `digitising_sigma_mm`: sqrt(rms^2 + half_line_width^2 + calibration^2),
  0.25-1.57 mm.
- `gnss_sigma_mm`: the unit's own scatter, 1.4826 x MAD of the daily values
  about the 7-day average (2.1-4.1 mm in Up, 1.1-2.4 mm in East and North),
  not divided by sqrt(7), as for the GNSS daily sigma of amendment
  2026-10-08/4.
- `sigma_up_mm` (and `sigma_mm`, `sigma_east_mm`, `sigma_north_mm`):
  sqrt(digitising_sigma^2 + gnss_sigma^2): 2.2-4.1 mm in Up. This is what
  `load_points_csv` gives, and the harness recomputes it from the terms
  above (`validation.digitised_truth_sigma`).
- As first fixed, kept for the harness's `version_1` report:
  `half_gridline_mm` (half the spacing of the plot's labelled gridlines: 50,
  25 or 10 mm) and `sigma_*_as_first_fixed_mm` = sqrt(max(rms,
  half_gridline)^2 + gnss_sigma^2), 10.2-50.1 mm in Up.

`tahmoor_west_gnss_calibration.json` gives each panel's terms
(`panels.*.sigma`). The rebuild that added them (8 Oct 2026) left the values,
dates, positions, the daily file, `checks/gnss_checks.json` and every overlay
byte for byte as they were.

**Positions.** Each unit's circle on the 1:10000 monitoring layout drawing
MSEC1263-01 (Report 8 page 88), which is tied by an affine fit to the LW W1-W3
corners read on the MGA-gridded drawings (12 corners, RMS 0.17 m). Each
figure's own inset map (LW W1-W4 and a red diamond) is tied the same way to the
W1-W4 corners; it picks the circle (the next circle is always more than 58 m
away) and agrees with it within 0.0-2.4 m. Stated sigma 5 m. Site 24 lies over
LW W4, 200 m from its north (commencing) end, as Report 8 says. Units on the
Picton-Mittagong Loop Line sit beside the survey pegs of their title chainage
(Site 11 beside peg ~89.69 km, title 89.629 km; checked by eye on a 500 dpi
render).

**Checks against independent numbers** (`checks/gnss_checks.json`):

- Report 7's plots (other axis scales, published 9 months earlier) against
  Report 8's at the shared epochs: RMS 0.01-0.9 mm, at most 6.7 mm (Site 8
  Up).
- Report 8 text, "additional ... mm of vertical subsidence between 1 January
  2023 and 30 June 2023": Site 7 printed 5, digitised 6.2; Site 8 printed 2,
  digitised 3.9; Site 24 printed 16, digitised 20.9; Site 9 printed 4,
  digitised -0.1 (Site 9's record has a gap from 6 Jan to 21 Feb 2023). These
  are a few mm, the size of each unit's daily scatter (3-4 mm), so they agree
  within 5 mm, not exactly; which day's reading MSEC took is not stated.
- Site 24 totals: Report 8 "approximately 660 mm" against 677 mm on its own
  plot at 30 Jun 2023; Report 7 "approximately 640 mm" against 647 mm at
  13 Nov 2022. The text's totals are below the plots by 17 and 7 mm. This is
  not rounding: 677 would round to 680. Report 8's 660 is about Report 7's 640
  plus Report 8's own +16 mm. The plot shows another 9 mm between 13 Nov and
  31 Dec 2022, which neither text counts. The text totals are approximate. The
  plots of the two reports agree with each other to 0.6 mm at Site 24, so the
  plots are the record used.
- The date axes, checked independently (review, `checks/review_checks.json`).
  Each figure draws the mining period as a pink bar. Put through pass A's date
  calibration, the bar's ends fall on the published dates in every figure:
  W1 start 15 Nov 2019, W3 start 13 Sep 2021, W4 start 16 May 2022 and W4 end
  13 Sep 2022. That is 33 ends, all within 0.04 day.

**Figure note.** Fig. G13 says the unit was disturbed in early January 2022
(printed "2002"), relocated on 28 March 2022 and "results normalised". Its
record has no data from 10 Jan to 29 Mar 2022; MSEC joined the later values to
the earlier ones, so any movement during the gap is not measured. Those rows
(31) carry a `note`. The position is the relocated one. Protocol A amendment
`2026-10-08/p23-4` (8 Oct 2026): the harness scores the rows from 2022-03-28
as a separate unit, `TW_G13b`, with its own datum (its first epoch,
2022-04-01), and never forms a change across the gap. The file keeps one id;
the split is the harness's (`validation.split_truth_units`).

**Not digitised.** Sites 14 and 23 (Report 5, Rev 08; Site 23 sat on the W3
centreline, about 345 mm). They are not among the 17 units and are not in
the protocol.

## 2. Panel outlines

**Sources.** The mines' CAD drawings in the extraction plans and end-of-panel
reports (`georef_drawings.json` lists each with its PDF, URL, page, title,
date and scale):

| drawing | report | panels read |
|---|---|---|
| MSEC1019-01 | Tahmoor W1-W2 Extraction Plan, App. A (MSEC1019 Rev B, Jun 2019), p200 | LW W1, LW W2 |
| MSEC1112-01 | Tahmoor W3-W4 Extraction Plan, App. A (MSEC1112 Rev A, Mar 2021), p180 | LW W3, LW W4 (planned), LW30, LW31, LW32 |
| MSEC1263-01 | Tahmoor West Report 8 (MSEC1263 Rev 21, Aug 2023), p88 | LW W4 as mined |
| MSEC1192-01 | Tahmoor South EP (MSEC1192 Rev A, May 2022), p245 | LW S1A |
| MSEC1050-01 | Appin LW902 End of Panel (MSEC1050 Rev A), p32 | LW901, LW902 |
| MSEC1176-01 | Appin LW903 EoP monitoring review, p37 | LW903 |
| MSEC1285-01 | Appin LW904 EoP monitoring review, p39 | LW904 |
| MSEC1334-01 | Appin LW905 EoP monitoring review, p41 | LW905 |
| MSEC975-01 | Appin LW707 monitoring report, p51 | LW707A, LW707B |
| MSEC1250-01 | Appin LW708 subsidence assessment, p44 | LW708A, LW708B |
| MSEC1394-01 | Appin LW709 subsidence assessment, p46 | LW709 |

**How.** Grid lines and panel outlines are vector paths. Each axis is fitted
to the drawing's MGA grid lines (residual 0.0-0.2 m); the value of the lines
was read by eye from zoomed renders. Two drawings have no grid (MSEC1263-01,
MSEC1285-01): they are tied by an affine fit to the corners of three panels
read on gridded drawings (RMS 0.17 and 0.18 m). The drawings say "Grid to MGA
co-ordinates" without the datum; they are converted as MGA94 (EPSG:28356).
GDA94 and GDA2020 differ by about 1.5-1.8 m here. The review's South32 check
(below) suggests that at least MSEC1334-01 (LW905, 2023) is in MGA2020. Its
LW905 matches South32's MGA2020 outline to 0.4 m as drawn, and to 1.7 m after
conversion from MGA94. That error is at most 1.5 m, well inside the 10 m sigma.

**Checks** (`panel_checks.json`):

- Every outline's length and width against the void dimensions printed in the
  report tables: within 2.8 m in length and 2.0 m in width (LW W4 as mined:
  722 m against "length of extraction 721 m", Report 7).
- The same mined panels on drawings whose grids were read separately agree to
  0.0-1.4 m (median edge offset; e.g. LW707A/B and LW708A/B on MSEC1394-01,
  LW901-903 and LW707A/B on MSEC1250-01, LW901-904 and LW709 on MSEC1334-01).
  A misread grid label would show as a whole grid step (500-2000 m).
- CA19 (GA: "located above longwall 902") lies inside LW902, 107 m from its
  edge.
- The "Picton Station" dot on MSEC1019-01 lies 11 m from Picton railway
  station's Wikipedia position (34.179088 S, 150.612575 E).
- A drawing by another hand, with a stated datum (review,
  `checks/review_checks.json`, `scripts/review_checks.py`). South32's
  APNSVY00237 (LW709-711 and LW905 Extraction Plan variation, 8 Feb 2023,
  1:6000, MGA2020) is fitted to its own grid. The grid spacing comes from the
  drawing's scale, and the line values from the labels. The fit is checked
  against the drawing's table of 24 application-area corners, which all lie
  within 2.5 m of the drawn outline. LW709 and LW905 from the MSEC drawings
  lie within 2.6 m of South32's voids under either datum reading. CA19 alone
  fixes Appin only north-south: LW902 is 2.2 km long, so a 1 km east-west
  grid misread would have kept CA19 inside it. This check also fixes Appin
  east-west.

**Uncertainty.** 10 m (1 sigma) for an outline from a gridded drawing (grid fit
< 0.2 m; edge placement and planned-versus-mined ends, e.g. LW904 stopped 2 m
short of its approved end); sqrt(10^2 + rms^2) for an affine transfer.

**Window status** (`in_window`): mined in 2017-11-01 .. 2022-12-31, or
completed in the six months before it and so still settling (LW30, completed
15 May 2017; LW901, 8 Sep 2017). LW707A (Aug 2016) is outside; the planned
LW W4 outline is kept but superseded. The six-month settling rule is mine, not
the protocol's.

**Dates and depths.** From the reports (each feature names its source):
Tahmoor North from the GeoTerra LW32 report table in the 2019 Annual Review
(LW30 20 Jun 2016 - 15 May 2017, LW31 28 Jun 2017 - 17 Aug 2018, LW32 29 Oct
2018 - 26 Sep 2019; the review's text also says LW32 ended on 22 and on 25 Sep
2019); Tahmoor West from Reports 1-8 (W1 15 Nov 2019 - 6 Nov 2020, W2 7 Dec
2020 - 17 Jun 2021, W3 13 Sep 2021 - 21 Mar 2022, W4 16 May - 13 Sep 2022);
Appin from the EoP reports and the drawings' start and finish labels. Draw
distance = tan(26.5 deg) x the largest depth of cover stated over the panel,
204-356 m. Protocol A says "0.5 x depth of cover at the panel".
tan(26.5 deg) = 0.4986, so the two differ by at most 1 m. Using the largest
stated depth is my reading of "at the panel"; it is the cautious choice for
both the exclusion polygon and the stable ring.

## 3. Survey lines (end-of-panel maxima)

`survey_lines.geojson`: the ARTC (railway) line over LW901 beside LW902 (LW902
EoP: 263 mm, page 9), the Menangle Road line, which runs diagonally over the
western end of LW903 and the eastern end of LW904, as the LW904 EoP Table 3-1
says (432 mm, LW903 EoP page 9; 649 mm, LW904 EoP summary page 10), and the
ARTC and M31 West lines over LW708B (990 and 813 mm, LW708 EoP summary pages
10 and 9). Each is the median position, in 20 m steps, of the line's survey
pegs (filled dots on the MSEC drawings); a hand-placed guide only says which
dots belong to the line. Median peg offset from the line 0.3-5.5 m (5.5 m on
the ARTC line, where embankment and cutting points form arrays beside the
track). Stated sigma 10 m. LW902's ARTC line runs over LW901, 35 m outside
LW902: that is why its 263 mm is below CA19's ~340 mm, as protocol A says.

## 4. The replacement deramp-exclusion polygon

`configs/experiments/context/appin_tahmoor_deforming.geojson`, built by
`scripts/deforming_build.py` with the APPROX file's rule: every panel mined or
settling in A's window (16 panels), buffered by its draw distance plus 50 m
(at least twice the outline sigma), merged into three areas (Appin 12.9 km2,
Tahmoor West and North 8.2 km2, Tahmoor South S1A 1.7 km2; 22.9 km2 in all,
against 185 km2 for the APPROX envelopes).

- CA19 is inside; CA07, PCTN, MENA and CRDX are outside (2.8, 1.2, 2.4 and
  16 km away).
- The pipeline's deramp-mask builder accepts it on a uniform grid of the
  processed crop: the fit keeps 86% of the crop (60% with APPROX;
  `tests/test_p23_digitise.py`). About 248 km2 of the box lies more than 2 km
  from it, against 74 km2 for APPROX (geometry, `checks/deforming_checks.json`).
- **The APPROX file misses parts of four panels' reach**: LW S1A (0.21 km2 of
  the panel itself, its reach up to 818 m outside the envelope), LW708B (0.13
  km2 of the panel, up to 700 m), LW707B and LW709 (draw only, up to 328 and
  126 m). All of it lies within the APPROX envelope + 2 km, so A1 and A3 as
  configured (deramp_buffer_m 2000 m, reference pixel more than 2 km away) are
  not affected. With the README's fallback `deramp_buffer_m: 1000` they would
  still be covered, by 182 m.
- The APPROX file is kept unchanged. In the phase 2/3 finishing pass (8 Oct
  2026, before any run, download or result) the A1, A2 and A3 configs were
  switched to the new polygon, and protocol A records it in a record-only
  amendment (`2026-10-08/a-deramp-polygon`); `site.deforming_area_approx` above
  the amendments is unchanged text. Changing it again once a run has started is a
  new run. The same configs read `panels.geojson`, `survey_lines.geojson`
  (event_polygons, event_lines) and A1/A3 the units CSV (validation_points).

## Rebuild

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate insar
cd ~/insar_pipeline/review_2026-10-08/truth/scripts
python sources.py            # sha256 + URL HEAD checks (network: HEAD requests only)
python panels_build.py       # panels.geojson, georef_drawings.json, panel_checks.json
python gnss_build.py         # the GNSS units, daily record, calibration, checks/
python deforming_build.py    # the deforming polygon, checks/deforming_checks.json
python survey_lines_build.py # survey_lines.geojson
python qc_map.py             # checks/qc_map.png
python review_checks.py      # checks/review_checks.json (independent review checks)
```

Renders go to `$T1_WORKDIR` (default `/tmp/t1_digitise`). A rebuild from an
empty render folder gives byte-identical outputs (checked 8 Oct 2026).
`pdfvec.py` reads a page's vector paths and words (poppler's `pdftocairo
-svg` and `pdftotext -bbox`); `drawgeo.py` holds the grid and affine fits;
`gnss_plot_vector.py` and `gnss_plot_raster.py` are the two passes.

## Independent review, 8 Oct 2026

What the reviewer changed, all in this folder or the group's test file:

- The Menangle Road line was named "west ends of LW903, LW904". The LW904 EoP
  Table 3-1 says the road crosses the western end of LW903 and the eastern
  end of LW904. The digitised geometry already did this: inside LW904 it lies
  on the eastern part of the panel. Only the name and this README were wrong.
  The name is fixed in `survey_lines_build.py`. A rebuild from an empty render
  folder changes nothing else.
- The wording on Site 24's totals ("the text rounds down") and on what the
  centred-mean test proves about dates was corrected. Two independent checks
  were added: `scripts/review_checks.py` writes `checks/review_checks.json`.
- Added the 1.35 deg grid-convergence note for E/N, and a note on the draw
  distance rule.

The reviewer also checked these and found them right:

- The sign of up (the plots say "+ve = subsidence").
- The printed line maxima (263, 432, 649, 990 and 813 mm on the cited pages).
- Site 24 on the W4 centreline, 201 m from the commencing (north) end. W4 was
  shortened at that end, so the as-mined and planned outlines share the south
  end. The W4 subsidence at Site 24 starts in mid-June 2022, as a face start
  200 m away implies.
- Site 9 lies 2 m and Site 8 21 m from the centreline the figure titles name.
- The S1A drawing's grid labels (6 208 000, 6 204 000, 280 000), read on a
  400 dpi render.
- Every in-window panel's reach (outline plus draw distance) lies inside the
  new polygon, and the stations lie outside it.

Left over from the first, cut-off attempt: a partial 5 MiB download,
`~/insar_pipeline/tahmoor_west_EP_app_A_subsidence_predictions_1.pdf`. Its
header says 22,288,681 bytes. The complete copy is
`sources/tahmoor_west_EP_app_A_subsidence_predictions_1.pdf`. Kyle to delete
the partial one:
`rm ~/insar_pipeline/tahmoor_west_EP_app_A_subsidence_predictions_1.pdf`.
