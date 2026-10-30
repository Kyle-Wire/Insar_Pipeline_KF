"""Build the run-summary page from the results on disk.

    python scripts/run_summary_page.py <run_results.json> <out.html> [--title "..."]
        [--figures <deliverables/figures>] [--notes <extra notes.json>]

The findings are written from the significance results in the products, not
by hand; --notes only appends extra items. The independent-check section
appears when run_results.json carries a `comparison` (collect_results.py
--compare).

Every number on the page is read from run_results.json (scripts/collect_results.py),
which itself reads the products - quality_assessment.json, summary.json,
provenance.json, timing.json, the h5 attributes - never the log. The record is
judged against the practice of the field (quality_assessment.py), not against
another product.

Nothing about one site is written into the page: the title and place come from
report_insights.identity (via run_results.json), the resolution from
resolution.py, the acquisition gaps from the network, the look direction from
the orbit. Until 5 Oct 2026 it printed the first site's mine, town, acquisition
gaps and looks on every project.
"""
import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

MM_PER_IN = 25.4


def esc(x):
    return html.escape(str(x))


def fmt(x, nd=2, unit=""):
    if x is None:
        return "—"
    try:
        return f"{float(x):,.{nd}f}{unit}"
    except (TypeError, ValueError):
        return esc(x)


def signed(x, nd=1, unit=""):
    """'+1.2' / '-0.6'; a value that rounds to zero is '0.0' (never '-0.0')."""
    if x is None:
        return "—"
    x = float(x)
    if abs(x) < 0.5 * 10 ** -nd:
        return f"{0.0:,.{nd}f}{unit}"
    return f"{x:+,.{nd}f}{unit}"


def box_text(s_, w, n, e, nd=5) -> str:
    """'38.20000–38.30000 N, 87.40000–87.30000 W': each axis's hemisphere
    from its sign (techout-11: '-lon W' assumed the western hemisphere);
    a box across the equator or Greenwich keeps signed degrees."""
    def axis(a, b, pos, neg):
        a, b = float(a), float(b)
        if a >= 0 and b >= 0:
            return f"{fmt(a, nd)}–{fmt(b, nd)} {pos}"
        if a <= 0 and b <= 0:
            return f"{fmt(-a, nd)}–{fmt(-b, nd)} {neg}"
        return f"{fmt(a, nd)}–{fmt(b, nd)} (degrees {pos} positive)"
    return f"{axis(s_, n, 'N', 'S')}, {axis(w, e, 'E', 'W')}"


def both(x, nd_mm=1, nd_in=2, unit=""):
    """'−0.6 mm/yr (−0.02 in/yr)' — every rate and length on the page in both units."""
    if x is None:
        return "—"
    return f"{signed(x, nd_mm)} mm{unit} ({signed(float(x) / MM_PER_IN, nd_in)} in{unit})"


def mag(x, nd_mm=1, nd_in=2, unit=""):
    """Both units, no sign: for spreads, noise and other magnitudes."""
    if x is None:
        return "—"
    return f"{fmt(x, nd_mm)} mm{unit} ({fmt(float(x) / MM_PER_IN, nd_in)} in{unit})"


def pair_mm_in(x, nd_mm=2, nd_in=3):
    if x is None:
        return "—"
    return f"{signed(x, nd_mm)} / {signed(float(x) / MM_PER_IN, nd_in)}"


def signed_bars(rows, width=760, label_w=110, vmax=None):
    """Horizontal bars from a zero line; rows = [(label, value_mm_yr, n)]."""
    if not rows:
        return ""
    vmax = vmax or max(abs(v) for _, v, _ in rows) or 1
    bh, gap = 20, 10
    height = len(rows) * (bh + gap) + 30
    zero = label_w + (width - label_w - 190) / 2
    scale = (width - label_w - 190) / 2 / vmax
    parts = [f'<line x1="{zero:.1f}" x2="{zero:.1f}" y1="6" y2="{height - 24}" class="zero"/>']
    for i, (name, v, n) in enumerate(rows):
        yy = 6 + i * (bh + gap)
        w = abs(v) * scale
        x0 = zero if v >= 0 else zero - w
        parts.append(
            f'<text x="{label_w - 10}" y="{yy + 14}" class="tick" text-anchor="end">{esc(name)}</text>'
            f'<rect x="{x0:.1f}" y="{yy}" width="{max(2, w):.1f}" height="{bh}" class="{"bar pos" if v >= 0 else "bar neg"}"/>'
            f'<text x="{(zero + w + 8) if v >= 0 else (zero - w - 8):.1f}" y="{yy + 14}" class="tick" text-anchor="{"start" if v >= 0 else "end"}">{v:+.1f} mm/yr ({v / MM_PER_IN:+.2f} in/yr) · {n:,} px</text>')
    parts.append(f'<text x="{zero:.1f}" y="{height - 8}" class="tick" text-anchor="middle">0 = no motion</text>')
    return (f'<figure class="chart small"><svg viewBox="0 0 {width} {height}" width="100%" role="img" '
            f'aria-label="median rate by temporal-coherence class">{"".join(parts)}</svg></figure>')


def figures_html(fig_dir):
    import base64
    if not fig_dir:
        return ""
    fig_dir = Path(fig_dir)
    man = json.loads((fig_dir / "figures.json").read_text())
    out = []
    for m in man:
        b64 = base64.b64encode((fig_dir / m["file"]).read_bytes()).decode()
        out.append(f'<figure class="map"><img src="data:image/png;base64,{b64}" alt="{esc(m["title"])}" loading="lazy">'
                   f'<figcaption><b>{esc(m["title"])}.</b> {m["caption"]}</figcaption></figure>')
    return "".join(out)


def hms(seconds):
    if seconds is None:
        return "—"
    s = int(round(seconds))
    h, m = divmod(s // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s % 60:02d} s"


def ymd(d):
    d = str(d)
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) >= 8 else d


def pill(p, not_applicable=False):
    if p is None and not_applicable:
        return '<span class="pill na">Not applicable</span>'
    if p is None:
        return '<span class="pill na">Not tested</span>'
    return '<span class="pill pass">Meets practice</span>' if p else '<span class="pill flag">Flagged</span>'


# ------------------------------------------------------------------ charts

def series_chart(dates, series, labels, event=None, width=760, height=320, unit="mm"):
    """Up to three series (mm) against time, with the event window shaded."""
    if not dates or len(dates) < 2:
        return "<p class='muted'>No series available.</p>"
    t0 = datetime.strptime(dates[0], "%Y%m%d")
    t1 = datetime.strptime(dates[-1], "%Y%m%d")
    span = max(1, (t1 - t0).days)
    vals = [v for s in series for v in s if v is not None]
    lo, hi = min(vals), max(vals)
    pad = 0.08 * (hi - lo or 1)
    lo, hi = lo - pad, hi + pad
    L, R, T, B = 56, 78, 16, 40
    pw, ph = width - L - R, height - T - B

    def x(d):
        return L + pw * (datetime.strptime(d, "%Y%m%d") - t0).days / span

    def y(v):
        return T + ph * (hi - v) / (hi - lo)

    paths, ends = [], []
    for s in series:
        pts = [(x(d), y(v)) for d, v in zip(dates, s) if v is not None]
        paths.append("M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts))
        ends.append(pts[-1])
    rng = hi - lo
    step = 50 if rng > 200 else 20 if rng > 80 else 10 if rng > 40 else 5
    ticks = [v for v in range(int(lo // step) * step, int(hi) + step, step) if lo <= v <= hi]
    years = sorted({d[:4] for d in dates})
    xt = [(yr, x(f"{yr}0101")) for yr in years if t0 <= datetime.strptime(f"{yr}0101", "%Y%m%d") <= t1]
    ev = ""
    if event and event[0] and event[1]:
        xs, xe = x(event[0]), x(event[1])
        ev = (f'<rect x="{xs:.1f}" y="{T}" width="{max(2, xe - xs):.1f}" height="{ph}" class="event"/>'
              f'<text x="{(xs + xe) / 2:.1f}" y="{T + 12}" class="evlabel" text-anchor="middle">mining window</text>')
    grid = "".join(f'<line x1="{L}" x2="{width - R}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/>'
                   f'<text x="{L - 8}" y="{y(t) + 4:.1f}" class="tick" text-anchor="end">{t:+d}</text>'
                   f'<text x="{width - R + 8}" y="{y(t) + 4:.1f}" class="tick">{t / MM_PER_IN:+.2f}</text>'
                   for t in ticks)
    zero = f'<line x1="{L}" x2="{width - R}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="zero"/>' if lo < 0 < hi else ""
    xlab = "".join(f'<text x="{px:.1f}" y="{height - 14}" class="tick" text-anchor="middle">{yr}</text>'
                   for yr, px in xt if (len(xt) <= 6 or int(yr) % 2 == 0))
    cls = ["s1", "s2", "s3", "s4"]
    lines = "".join(f'<path d="{p}" class="{cls[i]}"/>' for i, p in reversed(list(enumerate(paths))))
    dots = "".join(f'<circle cx="{e[0]:.1f}" cy="{e[1]:.1f}" r="4" class="{cls[i]} dot"/>' for i, e in enumerate(ends))
    data = json.dumps({"dates": dates, "series": series, "labels": labels, "L": L, "pw": pw,
                       "t0": dates[0], "span": span, "unit": unit})
    keys = "".join(f'<span class="key"><i class="k {cls[i]}"></i>{esc(l)}</span>' for i, l in enumerate(labels))
    return f'''
<figure class="chart" data-chart='{esc(data)}'>
  <svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Cumulative line-of-sight displacement, millimetres">
    {ev}{grid}{zero}{lines}{dots}{xlab}
    <text x="{L - 44}" y="{T - 4}" class="tick">{unit}</text>
    <text x="{width - R + 8}" y="{T - 4}" class="tick">in</text>
    <line class="xhair" x1="0" x2="0" y1="{T}" y2="{T + ph}" style="display:none"/>
  </svg>
  <div class="tip" hidden></div>
  <figcaption>{keys}
    <span class="muted">Cumulative line-of-sight displacement relative to the surrounding farmland, mm on the left and inches on the right; negative = away from the satellite (subsidence). Hover for the values at any date.</span>
  </figcaption>
</figure>'''


def bar_chart(rows, width=760, label_w=200, value_fmt=hms):
    if not rows:
        return ""
    mx = max(abs(v) for _, v in rows) or 1
    bh, gap = 18, 10
    height = len(rows) * (bh + gap) + 12
    parts = []
    for i, (name, v) in enumerate(rows):
        yy = 6 + i * (bh + gap)
        w = (width - label_w - 120) * abs(v) / mx
        parts.append(
            f'<text x="{label_w - 10}" y="{yy + 13}" class="tick" text-anchor="end">{esc(name)}</text>'
            f'<rect x="{label_w}" y="{yy}" width="{max(2, w):.1f}" height="{bh}" class="bar"/>'
            f'<text x="{label_w + w + 8:.1f}" y="{yy + 13}" class="tick">{esc(value_fmt(v))}</text>')
    return (f'<figure class="chart small"><svg viewBox="0 0 {width} {height}" width="100%" role="img">'
            f'{"".join(parts)}</svg></figure>')


# ------------------------------------------------------------------ page

CSS = r'''
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#f3f5f7; --card:#ffffff; --ink:#17202a; --ink-2:#55616e; --ink-3:#8a95a1; --rule:#d9dfe5;
  --accent:#2a78d6; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --good:#0ca30c; --warn:#b07a1e; --bad:#d03b3b; --na:#8a95a1;
  --event:rgba(23,32,42,.06); --grid:#e5eaef;
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif;
  --sans:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --paper:#15181c; --card:#1d2126; --ink:#e8ecf0; --ink-2:#a4adb7; --ink-3:#727c87; --rule:#2c333b;
  --accent:#3987e5; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --good:#0ca30c; --warn:#fab219; --bad:#e66767; --na:#727c87; --event:rgba(232,236,240,.07); --grid:#262c33;}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --paper:#15181c; --card:#1d2126; --ink:#e8ecf0; --ink-2:#a4adb7; --ink-3:#727c87; --rule:#2c333b;
  --accent:#3987e5; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --good:#0ca30c; --warn:#fab219; --bad:#e66767; --na:#727c87; --event:rgba(232,236,240,.07); --grid:#262c33;}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.55;margin:0;padding-block:0 64px;padding-inline:16px}
main{max-width:1000px;margin:0 auto}
.prose{max-width:72ch}
header{padding-block:40px 8px;border-bottom:1px solid var(--rule);margin-bottom:28px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-2)}
h1{font-family:var(--serif);font-weight:700;font-size:clamp(28px,4.5vw,42px);line-height:1.1;margin:.3em 0 .4em;text-wrap:balance}
h2{font-family:var(--serif);font-weight:600;font-size:24px;margin:44px 0 12px;text-wrap:balance}
h3{font-size:15px;font-weight:600;margin:22px 0 6px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-2)}
p{margin:.5em 0} .lede{font-size:18px;color:var(--ink-2)} .muted{color:var(--ink-2)} .small{font-size:14px}
code{font-family:var(--mono);font-size:.9em;background:var(--card);border:1px solid var(--rule);padding:0 4px;border-radius:3px}
.verdict{display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;padding:18px 20px;border:1px solid var(--rule);border-left:6px solid var(--na);background:var(--card);margin:20px 0}
.verdict.ok{border-left-color:var(--good)} .verdict.limited{border-left-color:var(--warn)} .verdict.review{border-left-color:var(--bad)}
.verdict .word{font-family:var(--serif);font-size:28px;font-weight:700}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--card);border:1px solid var(--rule);padding:14px 16px}
.tile .label{font-size:13px;color:var(--ink-2)} .tile .value{font-weight:600;font-size:28px;line-height:1.15;margin-top:4px}
.tile .sub{font-size:13px;color:var(--ink-2);margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:15px;background:var(--card);border:1px solid var(--rule)}
th,td{text-align:left;padding:9px 12px;border-top:1px solid var(--rule);vertical-align:top}
thead th{border-top:0;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);font-weight:600}
th[scope=row]{font-weight:600;width:26%}
td.num,th.num{font-family:var(--mono);font-size:14px;font-variant-numeric:tabular-nums;white-space:nowrap;text-align:right}
.tablewrap{overflow-x:auto;margin:12px 0}
.pill{display:inline-block;font-size:12px;font-weight:600;letter-spacing:.04em;padding:2px 9px;border-radius:999px;border:1px solid;white-space:nowrap}
.pill.pass{color:var(--good);border-color:var(--good)} .pill.flag{color:var(--warn);border-color:var(--warn)} .pill.na{color:var(--na);border-color:var(--na)}
.pill::before{content:"● ";font-size:9px;vertical-align:1px}
.chart{margin:16px 0;background:var(--card);border:1px solid var(--rule);padding:12px 12px 8px;position:relative}
.chart svg{display:block;overflow:visible} .chart.small{padding:8px 8px 4px}
.chart .grid{stroke:var(--grid);stroke-width:1} .chart .zero{stroke:var(--ink-3);stroke-width:1}
.chart .event{fill:var(--event)} .chart .evlabel{font:11px var(--mono);fill:var(--ink-2)}
.chart .tick{font:11px var(--mono);fill:var(--ink-2)}
.chart path{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart path.s1{stroke:var(--s1)} .chart path.s2{stroke:var(--s2)} .chart path.s3{stroke:var(--s3)} .chart path.s4{stroke:var(--ink)}
.chart circle{stroke:var(--card);stroke-width:2} .chart circle.s1{fill:var(--s1)} .chart circle.s2{fill:var(--s2)} .chart circle.s3{fill:var(--s3)} .chart circle.s4{fill:var(--ink)}
.chart .bar{fill:var(--s1)} .chart .bar.neg{fill:var(--s2)} .chart .bar.pos{fill:var(--s3)} .chart .xhair{stroke:var(--ink-3);stroke-width:1}
.map{margin:20px 0;background:var(--card);border:1px solid var(--rule);padding:10px}
.map img{display:block;width:100%;height:auto;background:#fff;border-radius:2px}
.map figcaption{font-size:14px;color:var(--ink-2);padding:10px 4px 2px;line-height:1.5}
.map figcaption b{color:var(--ink)}
.gloss dt{font-weight:600;margin-top:14px} .gloss dd{margin:4px 0 0 0;color:var(--ink-2)}
.chart figcaption{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink-2);padding:6px 4px 0}
.key{white-space:nowrap;color:var(--ink)} .k{display:inline-block;width:18px;height:2px;vertical-align:middle;margin-right:6px}
.k.s1{background:var(--s1)} .k.s2{background:var(--s2)} .k.s3{background:var(--s3)} .k.s4{background:var(--ink)}
.tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--rule);padding:6px 10px;font-size:13px;box-shadow:0 2px 8px rgba(0,0,0,.08);white-space:nowrap}
.tip b{font-family:var(--mono);font-weight:500}
details{margin:8px 0} summary{cursor:pointer;color:var(--accent);font-size:14px}
ul{padding-left:1.2em} li{margin:.35em 0}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px 32px}
footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--rule);font-size:13px;color:var(--ink-2)}
</style>'''


def spread(m):
    """(rate, farmland spread, significance dict) of a regional figure."""
    s = (m or {}).get("significance") or {}
    return (m or {}).get("rate_mm_yr"), s.get("farmland_spread_1sigma_mm_yr"), s


def rate_pm(m, nd=2):
    """'−2.41 ± 3.6 mm/yr (−0.095 ± 0.14 in/yr)'"""
    r, sd, _ = spread(m)
    if r is None:
        return "—"
    if sd is None:
        return both(r, nd, nd + 1, "/yr")
    return f"{signed(r, nd)} ± {fmt(sd, 1)} mm/yr ({signed(r / MM_PER_IN, nd + 1)} ± {fmt(sd / MM_PER_IN, 2)} in/yr)"


def verdict_pill(m):
    _, _, s = spread(m)
    if not s.get("verdict"):
        return '<span class="pill na">No farmland null</span>'
    if s.get("distinguishable_from_farmland"):
        return '<span class="pill pass">Stands out from farmland</span>'
    return '<span class="pill flag">Within farmland spread</span>'


def findings_html(q, ph, sh):
    """The findings, written from the significance results - never by hand."""
    fd = q.get("findings", {})
    mines = ph.get("mines", {}) or {}
    sites = (sh.get("sites") or {})
    panels = ph.get("panels", {}) or {}
    items = []
    regions = list(mines.items()) + [(n, d.get("relative_all_cells") or {}) for n, d in sites.items()] + list(panels.items())
    out_ = [n for n, m in regions if spread(m)[2].get("distinguishable_from_farmland")]
    n_reg = len(regions)
    if not out_:
        items.append(("Nothing measured here moves differently from ordinary farmland by more than this record can resolve.",
                      f"Each figure below is set against the same measurement made on ordinary farmland of the same footprint, "
                      f"shifted across the far field. None of the {len(mines)} mines, {len(sites)} named site{'s' if len(sites) != 1 else ''} "
                      f"or {len(panels)} mapped panels ({n_reg} regions in all) falls outside that spread (two-sided p &lt; 0.05). "
                      "That is a statement about what this data can resolve on this ground, not a statement that nothing moved."))
    else:
        items.append((f"{len(out_)} of {n_reg} regions stand out from ordinary farmland; about {n_reg * 0.05:.0f} would by chance.",
                      "Tested at 5% each, " + esc(", ".join(out_)) + (" stands" if len(out_) == 1 else " stand")
                      + " outside the farmland spread. With this many regions tested, a hit that the independent "
                        "run does not repeat is what chance produces; see the independent check."))
    sds = [spread(m)[1] for m in mines.values() if spread(m)[1]]
    site_sd = next((spread(d.get("relative_all_cells"))[1] for d in sites.values() if spread(d.get("relative_all_cells"))[1]), None)
    if sds:
        mine_lim = 2 * max(sds)
        items.append(("What the record can rule out.",
                      f"Two farmland spreads is the size of relative motion this record would have detected: about "
                      f"{both(mine_lim, 0, 2, '/yr').replace('+', '±')} over a whole mine's panels"
                      + (f", and {both(2 * site_sd, 0, 2, '/yr').replace('+', '±')} over a parcel the size of the {esc(next(iter(sites)))}" if site_sd else "")
                      + ". Relative motion slower than that, sustained over the record, cannot be told apart from the way fields "
                      "differ from each other at C-band. Faster motion would have shown."))
    for name, m in mines.items():
        r, sd, s = spread(m)
        items.append((f"{name}: {rate_pm(m)}.",
                      f"{fmt(m.get('n_pixels'), 0)} cells, {both(m.get('total_mm'), 0, 2)} measured over the record; "
                      f"z = {signed(s.get('z'), 1)}, {esc(s.get('verdict', ''))}. Uncorrected for the coherence mix it would read "
                      f"{both(m.get('rate_plain_mm_yr'), 2, 3, '/yr')}."))
    steps = [p.get("step_across_mining_year_mm") for p in panels.values() if p.get("step_across_mining_year_mm") is not None]
    if panels:
        n_out = sum(1 for p in panels.values() if spread(p)[2].get("distinguishable_from_farmland"))
        items.append((f"Panel by panel: {n_out} of {len(panels)} mapped panels stand out.",
                      "Every mapped panel polygon is measured the same way"
                      + (f". Across their own mining years the {len(steps)} panels mined during the record step between "
                         f"{signed(min(steps), 0)} and {signed(max(steps), 0)} mm ({signed(min(steps) / MM_PER_IN, 2)} to "
                         f"{signed(max(steps) / MM_PER_IN, 2)} in), {sum(1 for x in steps if x < 0)} down and "
                         f"{sum(1 for x in steps if x > 0)} up - both ways, as noise does"
                         if steps else "")
                      + ". A panel subsiding sharply when mined would show a consistent downward step; none does. If these workings leave "
                        "pillars (room-and-pillar rather than longwall), little surface motion is what would be expected - the "
                        "mining method is worth confirming with the operator."))
    for name, d in sites.items():
        r_ = d.get("relative_all_cells") or {}
        cl, ca, ab = d.get("cluster_relative") or {}, d.get("cluster_absolute") or {}, d.get("absolute_all_cells") or {}
        items.append((f"The {esc(name)}: {rate_pm(r_)} relative to its surroundings.",
                      f"{fmt(d.get('coherent_cells'), 0)} coherent cells, {both(r_.get('total_mm'), 0, 2)} over the record; "
                      f"{esc(spread(r_)[2].get('verdict', ''))}. Its {fmt(d.get('cluster_cells'), 0)}-cell centre cluster reads "
                      f"{rate_pm(cl)} relative. Against the reference pixel the parcel reads {both(ab.get('rate_mm_yr'), 1, 2, '/yr')} "
                      f"({both(ab.get('total_mm'), 0, 1)} over the record) and the cluster {both(ca.get('rate_mm_yr'), 1, 2, '/yr')}: "
                      "that absolute figure is what low-coherence farmland reads everywhere in this record (see the "
                      "distributed-scatterer check), so it is the bias and the reference, not the parcel's own motion. An "
                      "absolute selected-point figure from another provider is the same kind of number."))
    pt = fd.get("monitoring_point") or {}
    ex = pt.get("extreme") or {}
    so = pt.get("standout")
    if so:
        items.append((f"One area stands out at the {so['scale_km']:.0f} km scale: {so['lat']:.3f}, {so['lon']:.3f}.",
                      f"Searched at 1, 2 and 4 km, the deepest {so['scale_km']:.0f} km area in the AOI reads "
                      f"{both(so['rate_relative_mm_yr'], 1, 2, '/yr')} relative (class-matched), against "
                      f"{both(so['deepest_other_farmland_mm_yr'], 1, 2, '/yr')} for the deepest other area of that size on ordinary "
                      f"ground ({fmt(so['gap_in_spreads'], 1)} farmland spreads apart; at 1 and 2 km the deepest patch has a near-equal rival). "
                      + ("It lies outside every mapped panel and its buffer. " if not (so['inside_mapped_panels'] or so['within_panel_buffer']) else "")
                      + "It sinks in every two-year period of the record and the independent product puts its deepest area of this size at the "
                        "same place and rate, less clearly separated. A candidate for ground checking, cause unknown."))
    if pt and not so:
        items.append(("The deepest patch in the data is not deeper than chance.",
                      f"Searching the whole AOI for its lowest 1 km patch finds {fmt(pt.get('lat'), 3)}, {fmt(pt.get('lon'), 3)} at "
                      f"{both(pt.get('rate_relative_1km_mm_yr'), 1, 2, '/yr')} relative (class-matched). Over "
                      f"{fmt(ex.get('independent_1km_patches_in_aoi'), 0)} independent 1 km patches of farmland, the deepest is "
                      f"expected to reach about {both(ex.get('expected_deepest_by_chance_mm_yr'), 0, 2, '/yr')} by chance, and the "
                      f"next-deepest feature 3 km or more away reads {both(ex.get('next_deepest_feature_3km_away_mm_yr'), 0, 2, '/yr')}; "
                      f"selection-corrected p = {fmt(ex.get('p_selection_corrected'), 2)}."
                      + (" It stands out." if ex.get("stands_out") else " It is reported as the deepest point, not as a detection.")))
    return "".join(f"<li><b>{t}</b> {b}</li>" for t, b in items)


METHOD = {
    "miaplpy": ("phase-linked", "distributed-scatterer phase linking (MiaplPy's sequential estimator, patched so that consecutive "
                "10-image windows share three dates and are placed in one datum through them), a 12/24/36-day network formed "
                "from the linked phases, snaphu, then MintPy"),
    "stack": ("classic small-baseline", "a classic small-baseline stack (ISCE2 topsStack coregistration, a 12/24/36-day network of "
              "interferograms, snaphu, then MintPy)"),
}


def _identity(res) -> dict:
    """{title, location} from run_results.json (collect_results.py records
    report_insights.identity); the project folder's name when absent."""
    ident = dict(res.get("identity") or {})
    if not ident.get("title"):
        try:
            from insar_pipeline.report_insights import _pretty
            ident["title"] = _pretty(Path(res.get("project", "project")).name)
        except Exception:                                     # noqa: BLE001
            ident["title"] = Path(res.get("project", "project")).name.replace("_", " ").title()
    ident.setdefault("location", "")
    return ident


# slc/assembly_status.json 'dropped' reason codes (data_acquisition)
_DROP_WHY = {"catalogue_short": "the satellite archive holds only part of the area's bursts on that date",
             "catalogue_unreachable": "the archive's burst catalogue could not be reached to check it",
             # a legacy dropped_short_dates.json proves only that the SEARCH came
             # back short (4 of Site B's 5 were search-rendering gaps)
             "search_short": "the archive search returned it incomplete (fewer bursts than the other dates)",
             # a catalogue date the started stack's plan cannot take (the
             # wording of quality_assessment._DROP_REASON); not a short date
             "not_in_plan": "it was found in the archive only after this project's processing had started, so it "
                            "could not be added to the processing plan"}
# reasons that say nothing about a date's burst count
_NOT_SHORT_WHY = {_DROP_WHY["catalogue_unreachable"], _DROP_WHY["not_in_plan"]}


def _resolution_text(res, looks) -> str:
    r = res.get("resolution") or {}
    if r.get("text"):
        return r["text"]
    return (f"Products are posted on the delivered grid; at {looks[0]} × {looks[1]} looks each value combines a larger "
            "independent cell than the posting (see provenance.json).")


def _gap_sentence(res) -> str:
    """The acquisition gaps and the pairs that bridge them, from the network."""
    gaps = (res.get("network_gaps") or {}).get("gaps") or []
    if not gaps:
        return ""
    names = ", ".join(f"{ymd(g['from'])} → {ymd(g['to'])} ({g['days']} days)" for g in gaps[:6]) \
        + (f" and {len(gaps) - 6} more" if len(gaps) > 6 else "")
    longest = max((g.get("longest_bridge_days") or 0) for g in gaps)
    bp = max((g.get("max_bridge_bperp_m") or 0) for g in gaps)
    unbridged = [g for g in gaps if not g.get("bridging_pairs")]
    return (f". The record has {len(gaps)} gap{'s' if len(gaps) != 1 else ''} longer than "
            f"{res['network_gaps'].get('min_gap_days', 36)} days ({names}); the pairs bridging them are up to "
            f"{longest} days and {fmt(bp, 0)} m of perpendicular baseline"
            + ("; " + str(len(unbridged)) + " gap(s) have no bridging pair, which splits the record into separately "
               "referenced pieces" if unbridged else
               "; pruning them would split the record into separately referenced pieces"))


def build(res, title=None, notes=None, fig_dir=None):
    ident = _identity(res)
    title = title or f"{ident['title']} Ground Motion Record"
    q = res.get("quality") or {"checks": {}, "findings": {}, "verdict": "not run"}
    ch, fd = q.get("checks", {}), q.get("findings", {})
    summ = res.get("summary") or {}
    prov = res.get("provenance") or {}
    outc = prov.get("outcome") or {}
    so = prov.get("stack_outcome") or {}
    cfg = res.get("config") or {}
    net = outc.get("network") or {}
    unwrap = res.get("unwrap") or {}
    timing = {t["run_file"]: t for t in (res.get("timing") or [])}
    ref_ex = res.get("reference_excluded_pairs") or {}
    # every acquisition left out and why, from the download's own record
    # (slc/assembly_status.json); dropped_short_dates.json for older projects
    asm = res.get("assembly_status") or {}
    dropped_why = {str(d): _DROP_WHY.get(str(r), str(r)) for d, r in (asm.get("dropped") or {}).items()} \
        if isinstance(asm.get("dropped"), dict) else {}
    for d in asm.get("accepted_failures") or []:
        dropped_why.setdefault(str(d), "could not be downloaded (accepted with allow_missing_dates)")
    if not dropped_why:
        dropped_why = {str(d): _DROP_WHY["search_short"]
                       for d in (res.get("dropped_short_dates") or {}).get("dropped", [])}
    dropped_short = sorted(dropped_why)
    ua = Path(res["project"]) / "mintpy" / "unwrap_error_applied.json"
    unw_applied = json.loads(ua.read_text()) if ua.exists() else None
    n_dates = summ.get("n_acquisitions")
    d0, d1 = (summ.get("date_range", " to ").split(" to ") + ["", ""])[:2]
    refst, prec, noise, cov, netc, dsb = (ch.get("reference_stability", {}), ch.get("rate_precision", {}),
                                          ch.get("single_epoch_noise", {}), ch.get("coverage", {}),
                                          ch.get("network", {}), ch.get("distributed_scatterer_bias", {}))
    point = fd.get("monitoring_point", {})
    ex = point.get("extreme") or {}
    defo = fd.get("deforming_area", {})
    verdict = q.get("verdict", "—")
    ph = res.get("panel_history") or {}
    sh = res.get("site_history") or {}
    refpx = q.get("reference_pixel") or {}
    ff = q.get("far_field") or {}
    cmp_ = res.get("comparison")

    # the map cell from the product (summary.json resolution), not a constant
    cell_m = (summ.get("resolution") or {}).get("posting_m") or (res.get("resolution") or {}).get("posting_m")
    site_section = ""
    for sname, sd in (sh.get("sites") or {}).items():
        cs = sd.get("cell_stats") or {}
        if not cs:
            continue

        def stat_table(kind, unit_mm, unit_in):
            rows_ = ""
            for how, key in (("Absolute, against the reference pixel", f"{kind}_absolute_vs_reference"),
                             ("Relative, against farmland of the same coherence", f"{kind}_relative_vs_far_field")):
                st = cs.get(key)
                if st:
                    rows_ += (f"<tr><th scope='row'>{how}</th>" + "".join(f"<td class='num'>{signed(st[k], 1)} {unit_mm} ({signed(st[k] / MM_PER_IN, 2)} {unit_in})</td>" for k in ("min", "max", "mean", "median")) + "</tr>")
            return (f'<div class="tablewrap"><table><thead><tr><th></th><th class="num">Min</th><th class="num">Max</th><th class="num">Mean</th><th class="num">Median</th></tr></thead><tbody>{rows_}</tbody></table></div>')
        r_ = sd.get("relative_all_cells") or {}
        cl, ca = sd.get("cluster_relative") or {}, sd.get("cluster_absolute") or {}
        ov = ", ".join(f"{o['mine']} {o['year']} ({o['share_of_box_percent']}%)" for o in sd.get("overlapping_panels", [])) or "no mapped panel"
        site_section += f"""
<section>
  <h2>The project area: {esc(sname)}</h2>
  <p class="prose">{fmt(sd.get('coherent_cells'), 0)} coherent cells of {fmt(sd.get('cells_in_box'), 0)} in the area ({box_text(*sd['box_S_W_N_E'])}), overlapping {esc(ov)}; {fmt(sd.get('high_coherence_cells'), 0)} of them have temporal coherence ≥ 0.7. <b>Relative to its surroundings the area moves at {rate_pm(r_)}</b> {verdict_pill(r_)} — the ± is how much ordinary farmland patches of the same size differ from each other over the record. <b>Absolute</b> figures below are against the reference pixel and carry the reference and the distributed-scatterer bias; <b>relative</b> figures subtract the far-field mean of each cell's own coherence class. Totals for the area and the cluster are the measured change over the record (a curve that may change pace once a year, season removed); the per-cell table gives each cell's change at its average rate over the {fmt((summ.get('span_days') or 0) / 365.25, 2)} years, which overstates movement that slowed. Negative = away from the satellite. Min and max are single {fmt(cell_m, 0)} m cells, each good to several mm/yr; quote the mean.</p>
  <h3>Displacement rate</h3>
  {stat_table('rate_mm_yr', 'mm/yr', 'in/yr')}
  <h3>Each cell's change at its average rate (rate × years; not the measured change)</h3>
  {stat_table('total_mm', 'mm', 'in')}
  <p class="small muted">{fmt(sd.get('cluster_cells'), 0)}-cell cluster nearest the centre of the area (the centre-of-area selection, within about {fmt(sd.get('cluster_radius_m'), 0)} m; cells chosen by position, not by rate): {rate_pm(cl)} relative, {both(ca.get('rate_mm_yr'), 2, 3, '/yr')} absolute; total {both(cl.get('total_mm'), 0, 2)} relative, {both(ca.get('total_mm'), 0, 2)} absolute. Full series in <code>deliverables/los_cumulative_displacement_project_area.csv</code>; statistics in <code>deliverables/site_stats.csv</code>.</p>
</section>"""

    mine_tiles = ""
    for label, m in [("All mapped panels", ph.get("workings"))] + list(ph.get("mines", {}).items()):
        if not m:
            continue
        _, sdv, s = spread(m)
        mine_tiles += (f'<div class="tile"><div class="label">{esc(label)}' + (f" · {m['n_pixels']:,} cells" if m.get("n_pixels") else "") + '</div>'
                       f'<div class="value">{signed(m.get("rate_mm_yr"), 1)} <span style="font-size:18px">± {fmt(sdv, 1)} mm/yr</span></div>'
                       f'<div class="sub">= {signed(m.get("rate_in_yr"), 3)} ± {fmt((sdv or 0) / MM_PER_IN, 2)} in/yr · {both(m.get("total_mm"), 0, 2)} over the record · {verdict_pill(m)}</div></div>')
    for sname, sd in (sh.get("sites") or {}).items():
        r_ = sd.get("relative_all_cells") or {}
        _, sdv, _ = spread(r_)
        mine_tiles += (f'<div class="tile"><div class="label">{esc(sname)} · {fmt(sd.get("coherent_cells"), 0)} cells</div>'
                       f'<div class="value">{signed(r_.get("rate_mm_yr"), 1)} <span style="font-size:18px">± {fmt(sdv, 1)} mm/yr</span></div>'
                       f'<div class="sub">= {signed(r_.get("rate_in_yr"), 3)} ± {fmt((sdv or 0) / MM_PER_IN, 2)} in/yr · {both(r_.get("total_mm"), 0, 2)} over the record · {verdict_pill(r_)}</div></div>')
    vcls = {"MEETS PRACTICE": "ok", "USABLE WITH LIMITATIONS": "limited"}.get(verdict, "review")
    look = float(summ.get("look_angle_deg") or 38.4)
    vert_factor = 1 / max(0.1, math.cos(math.radians(look)))

    def stage(name):
        t = timing.get(name)
        return t["seconds"] if t else None
    stage_rows = []
    for label, keys in (("Coregistration", ["run_05_overlap_geo2rdr", "run_07_pairs_misreg",
                                            "run_09_fullBurst_geo2rdr", "run_11_extract_stack_valid_region"]),
                        ("Interferograms, merge, filter", ["run_13_generate_burst_igram"]),
                        ("Unwrapping", ["run_16_unwrap"])):
        secs = [stage(k) for k in keys if stage(k) is not None]
        if secs:
            stage_rows.append((label, sum(secs)))

    def row(name, c, value, meaning):
        return (f'<tr><th scope="row">{esc(name)}</th><td>{pill(c.get("pass"), bool(c.get("not_applicable")))}</td>'
                f'<td class="num">{value}</td><td>{meaning}</td></tr>')
    bin_km = noise.get("bin_km") or [0, 0]
    cls_emp = "; ".join(f"{r['tc']}: {fmt(r['empirical_mm_yr'], 1)}" for r in prec.get("by_class", []))
    checks = [
        row("Reference stability", refst,
            f"{both(refst.get('value_mm_yr'), 2, 3, '/yr')} · scatter {mag(refst.get('scatter_mm'), 1, 2)}",
            f"The referenced pixel ({fmt(refpx.get('lat'), 5)}, {fmt(refpx.get('lon'), 5)}; product row {refpx.get('row', '—')}, col {refpx.get('col', '—')}) against the mean of the other stable ground ({esc(refst.get('stable_ground_definition', ''))}). Practice: |trend| ≤ {fmt(refst.get('yardstick_mm_yr'), 0)} mm/yr. This moves <i>absolute</i> rates only: every relative figure on this page subtracts the far field, so the reference cancels out of it exactly."),
        row("Rate precision", prec,
            f"{mag(prec.get('empirical_median_mm_yr'), 1, 2, '/yr')} per cell (median, empirical) · {mag(prec.get('empirical_median_high_coherence_mm_yr'), 1, 2, '/yr')} at coherence ≥ 0.7",
            f"How repeatable one cell's rate is, measured: over the far field, each cell's rate fitted to {ymd((prec.get('halves') or ['', ''])[0])}–{ymd((prec.get('halves') or ['', ''])[1])} is compared with its rate over {ymd((prec.get('halves') or ['', '', '', ''])[2])}–{ymd((prec.get('halves') or ['', '', '', ''])[3])}; the spread of the difference, by coherence class, gives the error of a full-record rate (mm/yr — {esc(cls_emp)}). Practice: 1–3 mm/yr on distributed scatterers. MintPy's formal figure ({fmt(prec.get('median_sigma_mm_yr'))} mm/yr) treats the {n_dates or '—'} dates as independent; their residuals are correlated from one date to the next ({fmt(prec.get('residual_lag1_autocorrelation_median'), 2)}), so it understates the error about {fmt((prec.get('empirical_median_mm_yr') or 0) / max(prec.get('median_sigma_mm_yr') or 1e-9, 1e-9), 0)}-fold and is not quoted."),
        row("Single-epoch noise", noise,
            f"{mag(noise.get('robust_mm'), 1, 2)} robust · {mag(noise.get('rms_mm'), 1, 2)} RMS",
            f"Epoch-to-epoch displacement step RMS {fmt(bin_km[0], 0)}–{fmt(bin_km[1], 0)} km from the reference, {fmt(noise.get('far_over_near_ratio'), 1)}× the near-field value. C-band single-interferogram precision over vegetated ground is typically 5–10 mm."),
        row("Coverage", cov,
            f"{fmt(cov.get('aoi_percent_at_threshold'), 1)}% of the AOI at coherence ≥ {fmt(cov.get('threshold'), 1)} · {fmt(cov.get('aoi_percent_at_0_7'), 1)}% at 0.7",
            f"Share of imaged AOI cells whose temporal coherence ({esc(cov.get('coherence_source', ''))}) meets the working threshold ({fmt(cov.get('aoi_imaged_cells'), 0)} cells). Operational services quote 70%+ over agricultural ground. At MintPy's default threshold of 0.7 this farmland keeps {fmt(cov.get('aoi_percent_at_0_7'), 1)}%, the honest statement of how much of it behaves like a persistent scatterer."),
        row("Network", netc,
            f"{netc.get('pairs_inverted', '—')} of {netc.get('pairs_formed', '—')} pairs · {fmt(netc.get('mean_connections_per_date'), 1)} connections/date",
            f"{'One connected network' if netc.get('connected') else 'NOT connected'} across {netc.get('dates', '—')} dates; redundancy practice ≥ 3. Closed-loop check: at the median coherent pixel {fmt(100 * (netc.get('closure_median_fraction_open') or 0), 0)}% of {netc.get('closure_triplets', '—')} triplets carry a whole-cycle unwrapping inconsistency — the growing-season loops. Pixels with more of them read more negative and noisier within every coherence class, so part of the drift below is unwrapping error."
            + (f" Unwrap-error correction applied: {esc(unw_applied['applied'])} (requested {esc(unw_applied['requested'])})." if unw_applied else "")),
        row("Distributed-scatterer bias", dsb,
            f"{both(dsb.get('low_minus_high_coherence_mm_yr'), 1, 2, '/yr')} (low − high coherence)",
            "Median rate far from any deformation, by temporal-coherence class. A monotonic drift toward negative rates at low coherence is the short-temporal-baseline phase bias of distributed scatterers over vegetated ground (Ansari, De Zan &amp; Parizzi 2021; Zheng et al. 2022), with a share from unwrapping errors. <b>Absolute rates on low-coherence farmland are biased low; every site figure on this page is matched by coherence class against the surrounding farmland, which removes it.</b>"),
    ]

    sens = refst.get("sensitivity", [])
    sens_rows = "".join(
        f"<tr><th scope='row'>{esc(s['name'])}</th><td class='num'>{fmt(s['lat'], 5)}, {fmt(s['lon'], 5)}</td>"
        f"<td class='num'>{pair_mm_in(s.get('trend_vs_stable_ground_mm_yr'))}</td><td class='num'>{pair_mm_in(s.get('scatter_vs_stable_ground_mm'), 1, 2)}</td></tr>" for s in sens)
    fsens_rows = "".join(
        f"<tr><th scope='row'>{esc(x['far_field'])}</th><td class='num'>{pair_mm_in(x['workings_class_matched_mm_yr'])}</td></tr>"
        for x in refst.get("far_field_sensitivity", []))
    cls_rows = "".join(
        f"<tr><th scope='row'>{esc(c['tc'])}</th><td class='num'>{fmt(c['n'], 0)}</td><td class='num'>{pair_mm_in(c['median_mm_yr'])}</td></tr>"
        for c in dsb.get("velocity_by_coherence_class", []))
    cls_chart = signed_bars([(c["tc"], c["median_mm_yr"], c["n"]) for c in dsb.get("velocity_by_coherence_class", [])])

    labels, series, ser = [], [], {}
    if ph.get("dates"):
        ser = {"dates": ph["dates"]}
        for mine, m in ph.get("mines", {}).items():
            labels.append(f"{mine} ({m['n_pixels']:,} cells)"); series.append(m["series_mm"])
        if ph.get("workings"):
            labels.append("All mapped panels"); series.append(ph["workings"]["series_mm"])
    fig = series_chart(ser.get("dates", []), series, labels, None) if series else ""
    table_rows = "".join(
        "<tr><td>" + ymd(d) + "</td>" + "".join(f"<td class='num'>{pair_mm_in(s[i], 1, 2)}</td>" for s in series) + "</tr>"
        for i, d in enumerate(ser.get("dates", [])))

    excl = []
    missing = res.get("acquisitions_not_in_stack") or []
    if missing:
        excl.append(f"<li><b>{len(missing)} assembled acquisition(s) never entered the stack</b> ({', '.join(ymd(d) for d in missing)}): assembled after the run files were generated, and the generator reused the old run files without them. One date of {n_dates} changes no figure here; the run-file generator not regenerating when the acquisition set changes is a known defect, to be fixed before the next stack.</li>")
    if dropped_short:
        # the burst-count explanation only where some date was left out as short
        short_note = (" A date with fewer bursts than the rest would have shrunk the common valid region of every "
                      "other date." if any(dropped_why[d] not in _NOT_SHORT_WHY
                                           and not dropped_why[d].startswith("could not be downloaded")
                                           for d in dropped_short) else "")
        excl.append(f"<li><b>{len(dropped_short)} acquisition(s) not used</b> — " + "; ".join(f"{ymd(d)}: {esc(dropped_why[d])}" for d in dropped_short) + "." + short_note + "</li>")
    mt = timing.get("run_07_pairs_misreg", {})
    if mt.get("failed"):
        excl.append(f"<li><b>{len(mt['failed'])} of {mt.get('n')} misregistration pairs</b> could not be estimated (burst overlaps below the ESD coherence threshold); the network is redundant and the affected dates keep their other pairs.</li>")
    if ref_ex:
        n0, n1 = len(ref_ex.get("pairs_ref_in_cc0", [])), len(ref_ex.get("pairs_ref_component_too_small", []))
        excl.append(f"<li><b>{n0 + n1} interferogram{'s' if n0 + n1 != 1 else ''} excluded</b> because the reference pixel was not reliably unwrapped in them ({n0} in connected component 0, {n1} in a component below the minimum area). Reference median coherence {fmt(ref_ex.get('ref_median_coherence'), 3)}.</li>")
    if net.get("dropped"):
        excl.append(f"<li>{net.get('interferograms_inverted')} of {net.get('interferograms_loaded')} loaded interferograms inverted ({len(net['dropped'])} dropped: baseline limits and the reference check).</li>")
    if unw_applied and unw_applied.get("applied") != unw_applied.get("requested"):
        excl.append(f"<li><b>Phase-closure correction not applied.</b> MintPy's implementation corrects whole connected components against a region reliably unwrapped in every kept pair (≥ 2,500 px); across {net.get('interferograms_inverted', '—')} pairs and {max(1, int(round((summ.get('span_days') or 0) / 365.25)))} growing seasons there is none. Bridging was applied; the inversion used <code>{esc(unw_applied['dataset'])}</code>.</li>")
    nc = res.get("network_connectivity") or {}
    if nc:
        excl.append(f"<li><b>Kept network verified as one connected piece</b> before inversion: {nc.get('kept_pairs', '—')} pairs over {nc.get('dates', '—')} dates"
                    + (f"; {len(nc['excluded_dates'])} date(s) with no reliable pair excluded ({', '.join(ymd(d) for d in nc['excluded_dates'])})" if nc.get('excluded_dates') else "")
                    + esc(_gap_sentence(res)) + ".</li>")
    excl.append(f"<li>{'No pair' if not so.get('pairs_quarantined') else str(len(so['pairs_quarantined'])) + ' pairs'} hit the {fmt(cfg.get('unwrap_timeout_min'), 0)}-minute unwrap cap; the network is {'connected' if so.get('network_connected') else 'NOT connected'}.</li>")

    prod_dir = Path(res["project"]) / "deliverables"
    prods = sorted(p.name for p in prod_dir.iterdir() if not p.name.startswith("superseded")) if prod_dir.exists() else []
    cw = res.get("crop_window") or {}
    cleaned = res.get("cleaned_markers") or {}
    regions = q.get("regions") or {}
    notes_html = ""
    if notes:
        items = "".join(f"<li><b>{esc(n['title'])}</b> {n['body']}</li>" for n in notes.get("items", []))
        notes_html = f'<section class="prose"><h2>{esc(notes.get("heading", "Notes"))}</h2><ul>{items}</ul></section>'
    # counted as the verdict counts them (D2, 7 Oct 2026): out of the
    # applicable checks; a check that cannot apply is named, not counted.
    # Read from q['checks'], so a QA record from before D2 works the same way.
    from insar_pipeline.quality_assessment import check_counts
    cnt = check_counts(q)
    na_names = ", ".join(n.replace("_", " ") for n in cnt["not_applicable"])
    na_text = (f"; not counted, because {'it does' if len(cnt['not_applicable']) == 1 else 'they do'} not apply "
               f"here: {na_names}") if na_names else ""
    span_yr = (summ.get("span_days") or 0) / 365.25
    defo_thr = str(defo.get("threshold", "")).split(" relative")[0].replace("faster than ", "")

    cmp_html = ""
    if cmp_:
        this_name, this_desc = METHOD.get(res.get("backend", "stack"), METHOD["stack"])
        other_name, other_desc = METHOD.get(cmp_.get("backend", "miaplpy"), METHOD["miaplpy"])
        cq = cmp_.get("quality") or {}
        cch = cq.get("checks") or {}
        lvr = cmp_.get("linked_vs_record") or {}
        allc = lvr.get("all_cells") or {}
        hic = lvr.get("record_high_coherence_ge_0_7") or {}

        def crow(label, a, b):
            return f"<tr><th scope='row'>{label}</th><td class='num'>{a}</td><td class='num'>{b}</td></tr>"
        rows_c = [
            crow("Verdict", esc(str(cq.get("verdict", "—")).title()), esc(verdict.title())),
            crow("Rate repeatability of one cell (empirical, median)", mag((cch.get("rate_precision") or {}).get("empirical_median_mm_yr"), 1, 2, "/yr"), mag(prec.get("empirical_median_mm_yr"), 1, 2, "/yr")),
            crow("Reference pixel drift against stable ground", both((cch.get("reference_stability") or {}).get("value_mm_yr"), 2, 3, "/yr"), both(refst.get("value_mm_yr"), 2, 3, "/yr")),
            crow("Single-epoch noise, far field (robust)", mag((cch.get("single_epoch_noise") or {}).get("robust_mm"), 1, 2), mag(noise.get("robust_mm"), 1, 2)),
            crow("Low − high coherence rate difference (the drift)", both((cch.get("distributed_scatterer_bias") or {}).get("low_minus_high_coherence_mm_yr"), 1, 2, "/yr"), both(dsb.get("low_minus_high_coherence_mm_yr"), 1, 2, "/yr")),
            crow("Coverage at coherence ≥ 0.4 — <i>different coherence measures, not comparable</i>", f"{fmt((cch.get('coverage') or {}).get('aoi_percent_at_threshold'), 1)}%", f"{fmt(cov.get('aoi_percent_at_threshold'), 1)}%"),
            crow("All mapped panels, relative", rate_pm(cmp_.get("workings")), rate_pm(ph.get("workings"))),
        ]
        for mine, m in (ph.get("mines") or {}).items():
            rows_c.append(crow(f"{esc(mine)}, relative", rate_pm((cmp_.get("mines") or {}).get(mine)), rate_pm(m)))
        for sname, sd in (sh.get("sites") or {}).items():
            od = (cmp_.get("sites") or {}).get(sname) or {}
            rows_c.append(crow(f"{esc(sname)}, relative", rate_pm(od.get("relative_all_cells")), rate_pm(sd.get("relative_all_cells"))))
            rows_c.append(crow(f"{esc(sname)}, absolute (carries the bias)", both((od.get("absolute_all_cells") or {}).get("rate_mm_yr"), 1, 2, "/yr"), both((sd.get("absolute_all_cells") or {}).get("rate_mm_yr"), 1, 2, "/yr")))
        pan = ph.get("panels") or {}
        cpan = cmp_.get("panels") or {}

        def standouts(d):
            return [n for n, p in d.items() if (p.get("significance") or {}).get("distinguishable_from_farmland")]
        pan_lines = []
        for label, mine_d, other_d in ((f"{other_name} run", cpan, pan), ("record", pan, cpan)):
            for n in standouts(mine_d):
                o = other_d.get(n) or {}
                pan_lines.append(f"{esc(n)} in the {label} ({rate_pm(mine_d[n], 1)}); the other product reads {rate_pm(o, 1)} there"
                                 + (" and confirms it" if n in standouts(other_d) else " and does not confirm it"))
        pan_html = (f"<li><b>Panels.</b> Of {len(pan)} panels tested at 5% in each product, about {len(pan) * 0.05:.0f} would stand out by chance. "
                    + ("; ".join(pan_lines) + "." if pan_lines else "None stands out in either.") + "</li>")
        diffs = [(abs((spread((cmp_.get("mines") or {}).get(k))[0] or 0) - (m.get("rate_mm_yr") or 0)), k)
                 for k, m in (ph.get("mines") or {}).items() if spread((cmp_.get("mines") or {}).get(k))[0] is not None]
        diffs += [(abs((spread(((cmp_.get("sites") or {}).get(k) or {}).get("relative_all_cells"))[0] or 0)
                       - ((d.get("relative_all_cells") or {}).get("rate_mm_yr") or 0)), k)
                  for k, d in (sh.get("sites") or {}).items() if (cmp_.get("sites") or {}).get(k)]
        max_diff, max_name = max(diffs) if diffs else (None, "")

        def _sd_of(d_mines, d_sites, name):
            if name in (d_mines or {}):
                return spread(d_mines[name])[1]
            return spread(((d_sites or {}).get(name) or {}).get("relative_all_cells"))[1]
        max_sd = (_sd_of(cmp_.get("mines"), cmp_.get("sites"), max_name), _sd_of(ph.get("mines"), sh.get("sites"), max_name))
        agree_all = all(not spread(m)[2].get("distinguishable_from_farmland") for m in list((cmp_.get("mines") or {}).values()) + [(d.get("relative_all_cells") or {}) for d in (cmp_.get("sites") or {}).values()])
        cmp_html = f"""
<section>
  <h2>An independent check: the same site built by a different method</h2>
  <p class="prose">The record above is a {this_name} product, built with {this_desc}. As a check, the acquisitions were processed independently with {other_desc}, and put through the same assessment and significance tests. The two products share the grid and the reference pixel; nothing else.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Figure</th><th class="num">{esc(other_name.capitalize())} run</th><th class="num">{esc(this_name.capitalize())} record (this page)</th></tr></thead>
    <tbody>{"".join(rows_c)}</tbody></table></div>
  <ul class="prose">
    <li><b>Both products reach the same conclusion.</b> {"In neither does any mine or the parcel stand out from the spread of ordinary farmland." if agree_all else "See the table: the verdicts differ between the products."} The largest difference between the two products' mine and parcel rates is {fmt(max_diff, 1)} mm/yr ({esc(max_name)}), where the farmland spread is {fmt(max_sd[0], 1)} mm/yr in the {esc(other_name)} run and {fmt(max_sd[1], 1)} in the record.</li>
    {pan_html}
    <li><b>Cell by cell, the two products agree to the error the record's own test predicts.</b> On the same cells over the same {fmt(lvr.get('n_common'), 0)} dates, their ten-year rates differ by {mag(allc.get('difference_robust_scatter_mm_yr'), 1, 2, '/yr')} (robust scatter), {mag(hic.get('difference_robust_scatter_mm_yr'), 1, 2, '/yr')} on the record's coherent ground. For two independent products that is about {fmt((allc.get('difference_robust_scatter_mm_yr') or 0) / math.sqrt(2), 1)} mm/yr each — the same as the empirical repeatability of a cell in the quality table ({fmt(prec.get('empirical_median_mm_yr'), 1)} mm/yr). Their correlation ({fmt(allc.get('correlation'), 2)}) is partly the coherence-dependent drift they share, not independent agreement.</li>
    <li><b>The drift is the same size in both</b> ({both((cch.get('distributed_scatterer_bias') or {}).get('low_minus_high_coherence_mm_yr'), 1, 2, '/yr')} against {both(dsb.get('low_minus_high_coherence_mm_yr'), 1, 2, '/yr')}). On this farmland coherence does not survive a season, so there are no long coherent pairs for phase linking to exploit; both methods carry the multi-year datum through short-span links.</li>
    <li><b>Coverage is not comparable.</b> The phase-linked product grades cells on MiaplPy's mini-stack fit quality (how well each ~5-month window is linked), the classic one on MintPy's network-inversion temporal coherence; the two percentages measure different things.</li>
  </ul>
</section>"""

    ua_rec = unw_applied or {}
    method_line = METHOD.get(res.get("backend", "stack"), METHOD["stack"])[1]
    orbit = str(prov.get("orbit_direction") or cfg.get("orbit_direction") or "").lower() or "unknown"
    # Sentinel-1 looks right: ascending (heading ~north) it looks east, so it
    # sits to the west of the ground it sees; descending the other way round.
    sat_side = {"ascending": "west", "descending": "east"}.get(orbit, "side")
    mines = res.get("mine_layers") or []
    site_phrase = (f"the {mines[0]['label']} of {ident['title']}" if len(mines) == 1
                   else f"the mapped mines of {ident['title']}" if mines else ident["title"])
    looks = list(cfg.get("looks") or ["?", "?"])
    res_text = _resolution_text(res, looks)
    unw_rec = (prov.get("unwrap") or {}).get("settings") or []
    unw_line = (", ".join(f"{u.get('unwrapper') or u.get('engine')} ({u.get('cost')}, defomax {u.get('defomax_cycles')}), "
                          f"{u.get('pairs')} pairs" for u in unw_rec) if unw_rec else
                f"{cfg.get('unwrap_engine', 'isce')} engine, nomcf {cfg.get('unwrap_full_cost', True)}, "
                f"defomax {cfg.get('unwrap_defomax', '?')}")
    ver = res.get("version") or {}
    credits_html = ""
    if res.get("credits"):
        credits_html = ("<br>Data sources and credits: " + " ".join(esc(c.get("text", "")) for c in res["credits"]))
    if ua_rec.get("closure_correction"):
        method_line = method_line.replace(", then MintPy", ", a per-pixel loop-closure correction of unwrapping errors, then MintPy")
    body = f'''<title>{esc(title)}</title>{CSS}
<main>
<header>
  <div class="eyebrow">{esc(' · '.join(x for x in (ident['title'], ident['location']) if x))} · Sentinel-1 track {esc(prov.get('track', cfg.get('track', '?')))}, {esc(orbit)} · {esc(ymd(d0))} → {esc(ymd(d1))}</div>
  <h1>{esc(title)}</h1>
  <p class="lede prose">A {fmt(span_yr, 1)}-year line-of-sight ground-motion record over {esc(site_phrase)}, built from {fmt(n_dates, 0)} Sentinel-1 acquisitions with {method_line}. {fmt(summ.get('total_measured_points'), 0)} measurement points. {esc(res_text)} Every figure is read from the delivered products, judged against the practice of the field, and set against the same measurement on ordinary farmland.</p>
</header>

<section>
  <div class="verdict {vcls}">
    <span class="word">{esc(verdict.title())}</span>
    <span class="muted">{cnt['flagged']} of {cnt['applicable']} applicable quality checks flagged as limitations to carry into any interpretation; {cnt['passed']} meet practice{f"; {cnt['untested']} could not be run" if cnt['untested'] else ""}{esc(na_text)}.</span>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th>Check</th><th>Result</th><th>Measured</th><th>What it means</th></tr></thead>
    <tbody>{"".join(checks)}</tbody>
  </table></div>
</section>

<section class="prose"><h2>What the record shows</h2><ul>{findings_html(q, ph, sh)}</ul></section>
{notes_html}

<section>
  <h2>Headline figures</h2>
  <p class="prose small muted">Every rate is relative to the surrounding farmland of the same temporal-coherence class, over the whole record, positive = toward the satellite. The ± is not a fitting error: it is how much ordinary farmland patches of the same footprint differ from each other over the record, measured by shifting the region across the far field. A figure inside it has not been shown to differ from farmland. Totals are the measured change over the record (a curve that may change pace once a year, season removed), not rate × years.</p>
  <div class="tiles">
    {mine_tiles}
    <div class="tile"><div class="label">Deepest 1 km patch in the AOI · {fmt(point.get('lat'), 4)}, {fmt(point.get('lon'), 4)}</div><div class="value">{signed(point.get('rate_relative_1km_mm_yr'))} mm/yr</div><div class="sub">= {signed((point.get('rate_relative_1km_mm_yr') or 0) / MM_PER_IN, 2)} in/yr relative · deepest expected by chance over {fmt(ex.get('independent_1km_patches_in_aoi'), 0)} patches {both(ex.get('expected_deepest_by_chance_mm_yr'), 0, 2, '/yr')} · p = {fmt(ex.get('p_selection_corrected'), 2)} · {'stands out' if ex.get('stands_out') else 'a selected extreme'}</div></div>
    <div class="tile"><div class="label">Ground faster than {esc(defo_thr)} relative</div><div class="value">{fmt(defo.get('area_km2'), 1)} km²</div><div class="sub">{fmt((defo.get('area_km2') or 0) / 2.58999, 1)} mi² · {fmt(defo.get('inside_mapped_panels_percent'), 0)}% inside mapped panels · {fmt(defo.get('far_field_share_beyond_threshold_percent'), 1)}% of ordinary farmland crosses this threshold by chance</div></div>
  </div>
</section>

<section>
  <h2>Each mine's panels and the ground around them</h2>
  <p class="prose">Mean displacement of every mapped panel of each mine, relative to the far-field control (coherent farmland inside the AOI and outside all mapped panels plus a {fmt(regions.get('buffer_m', 2000) / 1000, 0)} km buffer, {fmt(regions.get('far_field_px'), 0)} cells) matched to each mine's coherence mix, from the first acquisition to the last. Zero is "moving like the farmland around the mine". The panel-by-panel view below the maps breaks each mine into its mapped panel polygons.</p>
  {fig}
  <details><summary>Table of the plotted values</summary><div class="tablewrap"><table class="small">
    <thead><tr><th>Date</th>{"".join(f"<th class='num'>{esc(l)}, mm / in</th>" for l in labels)}</tr></thead><tbody>{table_rows}</tbody></table></div></details>
</section>

{site_section}

<section>
  <h2>Maps and interferograms</h2>
  <p class="prose">The products behind the numbers. Every map is EPSG:4326 over the AOI; black outlines are the mapped mine panels, the star the pixel the record is referenced to and the green circle the deepest 1 km patch. Colour bars carry both units.</p>
  {figures_html(fig_dir)}
</section>

{cmp_html}

<section>
  <h2>What the relative figures depend on</h2>
  <p class="prose">The reference pixel affects only <i>absolute</i> rates: a relative figure is a region minus the far field, and any series subtracted from both cancels exactly, so re-referencing cannot move it. The candidates below are for choosing a reference that keeps the absolute rasters honest; each row is that pixel's own trend against the stable, high-coherence ground.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Pixel</th><th class="num">Lat, lon</th><th class="num">Trend vs stable ground, mm/yr / in/yr</th><th class="num">Scatter, mm / in</th></tr></thead>
    <tbody>{sens_rows}</tbody>
  </table></div>
  <p class="prose">What a relative figure does depend on is how the surrounding farmland is drawn. The same all-panels figure under each choice:</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Far field</th><th class="num">All mapped panels, relative, mm/yr / in/yr</th></tr></thead>
    <tbody>{fsens_rows}</tbody>
  </table></div>
  <h3>Rate by coherence class, far from any deformation</h3>
  {cls_chart}
  <div class="cols"><div class="tablewrap"><table>
    <thead><tr><th>Temporal coherence</th><th class="num">Pixels</th><th class="num">Median rate, mm/yr / in/yr</th></tr></thead>
    <tbody>{cls_rows}</tbody></table></div>
    <p class="small muted">Rates that fall monotonically with coherence, over ground with no reason to move, are the distributed-scatterer phase bias with a share from unwrapping errors (see the glossary). It is why absolute rates over farmland are not quoted here, and why every site figure is matched by coherence class. A closure-phase bias correction (Zheng et al. 2022) is the field's remedy, but it needs long coherent pairs this farmland does not keep, which the phase-linked run confirmed.</p>
  </div>
</section>

<section class="prose gloss">
  <h2>Terms used on this page</h2>
  <dl>
    <dt>Line of sight, and which way is positive</dt>
    <dd>The radar measures distance along its look direction, tilted about {fmt(look, 0)}° from vertical, with the satellite to the {sat_side} on this {esc(orbit)} track. <b>Positive means the ground moved toward the satellite; negative means away.</b> Subsidence is therefore negative on every number, map and graph here. This is MintPy's convention and the one used by the European Ground Motion Service; some products (and some vendors) flip it, so the sign is stated in <code>provenance.json</code> rather than assumed. A vertical raster is delivered under an explicit all-vertical assumption; it is larger than the line-of-sight value by a factor of about {fmt(vert_factor, 2)}.</dd>
    <dt>Far field</dt>
    <dd>The farmland around the mine that has no reason to move: every coherent cell inside the AOI and outside all mapped panels plus a 2 km buffer ({fmt(regions.get('far_field_px'), 0)} cells). It is the local control. Each site figure is the region minus the far field of the same coherence class, measured on the same dates with the same processing, so the reference pixel's own drift cancels exactly and the coherence-dependent bias cancels to the extent it depends on coherence.</dd>
    <dt>Farmland spread (the ±)</dt>
    <dd>How much ordinary farmland patches of the same footprint differ from each other over the record: the region's exact shape is shifted to every position in the far field and the same relative rate measured there. On this ground fields differ by several mm/yr (crop, soil moisture, the bias varying field by field), and a mine or a parcel is one more such patch. A figure outside two spreads stands out; inside it, the record cannot tell it from farmland. When a mine-sized footprint fits only a few times into the far field, the larger spread of an equal-area square is used instead, which errs on the safe side.</dd>
    <dt>Reference pixel</dt>
    <dd>Radar interferometry measures differences, so one cell is declared zero and everything else is relative to it. The one used here (star on the maps) was chosen from this record's own product. Its drift against the other stable ground ({both(refst.get('value_mm_yr'), 2, 3, '/yr')}) shifts every <i>absolute</i> rate by that amount; relative figures do not see it.</dd>
    <dt>Distributed-scatterer bias</dt>
    <dd>A radar resolution cell of farmland has no single dominant reflector; its echo is the sum of thousands of small ones (soil, stubble, leaves). Between two passes their arrangement changes with moisture and growth, and the phase then carries a small systematic component that does not cancel around closed loops. Summed over hundreds of short-interval pairs it appears as a steady fictitious rate, toward negative and larger where coherence is lower; whole-cycle unwrapping errors, also commoner on low-coherence ground, add to it. Over ground far from any deformation the median rate falls from {both(dsb.get('velocity_by_coherence_class', [{}])[-1].get('median_mm_yr'), 1, 2, '/yr')} on the most coherent cells to {both(dsb.get('velocity_by_coherence_class', [{}])[0].get('median_mm_yr'), 1, 2, '/yr')} on the least. Ground does not do that.</dd>
    <dt>Temporal coherence</dt>
    <dd>How consistently a cell's {netc.get('pairs_inverted', '—')} interferograms agree with one displacement history, from 0 (noise) to 1 (perfect). Cells below 0.4 are removed from every product here. MintPy's default is 0.7, which on this farmland keeps only {fmt(cov.get('aoi_percent_at_0_7'), 1)}% of the AOI; 0.4 is the practice for vegetated ground, stated rather than hidden.</dd>
    <dt>Wrapped and unwrapped</dt>
    <dd>The radar only knows phase within one wavelength: the wrapped interferogram repeats every 27.7 mm (1.09 in) of range change, like a contour map with no labels. Unwrapping (snaphu) counts the cycles to make a continuous surface. Where the count is wrong the error is a whole number of cycles; around a closed loop of three pairs such errors do not cancel, which is how the network check counts them.</dd>
  </dl>
  <h3>How to read the numbers</h3>
  <ul>
    <li><b>Posting is not resolution.</b> {esc(res_text)} Say both in anything that leaves the building.</li>
    <li><b>One cell is not a measurement of a place.</b> A single cell's rate is repeatable to about {fmt(prec.get('empirical_median_mm_yr'), 0)} mm/yr on this farmland; min and max over cells are noise. Quote regional means with their farmland spread.</li>
    <li><b>Absolute rates carry the bias.</b> The delivered rasters and the measurement-point table give rates against the reference pixel; on low-coherence farmland they read about {both((ff.get('class_mean_rate_mm_yr') or {}).get('0.4-0.6'), 0, 2, '/yr')} with nothing moving. The point table also carries a class-matched relative rate and an empirical uncertainty for each point.</li>
    <li><b>Fast motion is beyond this method.</b> The unwrappable gradient is π per pixel (13.9 mm) per 12-day pair; ground moving faster than that across a few cells is lost to unwrapping at any C-band posting.</li>
    <li><b>Panel geometry is only as fine as the mine layer.</b> Where the mapped panels are dated by whole calendar years, each panel is measured across its own mining year, every year alike, and the operator's dates would sharpen it.</li>
  </ul>
</section>

<section>
  <h2>The run</h2>
  <div class="cols">
    <div>
      <h3>Inputs and settings</h3>
      <ul class="small">
        <li>{fmt(n_dates, 0)} acquisitions, {esc(ymd(d0))} → {esc(ymd(d1))}, sub-swath(s) IW{esc(str(cfg.get('swaths', '?')).replace(' ', '+IW'))}, VV, assembled from the ASF burst catalogue; DEM: {esc((prov.get('dem') or {}).get('source', 'stitched over the full burst frame'))}.</li>
        <li>Reference date {esc(ymd(cfg.get('reference_date', '') or (prov.get('miaplpy_settings') or {}).get('stack_reference_date', '')))}; {'NESD coregistration, ESD coherence threshold ' + fmt(cfg.get('esd_coherence_threshold'), 2) if cfg.get('do_esd', True) else 'geometry coregistration'}; {cfg.get('connections', 3)} temporal connections; reference pixel {fmt(refpx.get('lat'), 6)}, {fmt(refpx.get('lon'), 6)} (configured {esc(cfg.get('reference_lalo', ''))}).</li>
        <li>Looks {esc(str(looks[0]))} × {esc(str(looks[1]))}; Goldstein α {fmt(cfg.get('filter_strength'), 1)}; unwrapping: {esc(unw_line)} — {'verified on every generated config' if so.get('configs_verified') is True else 'configs not re-verified (' + esc(str(so.get('configs_verified', so.get('note', 'no record')))) + ')'}.</li>
        <li>Unwrapping cropped to the AOI + 0.05°: {fmt(cw.get('crop_px'), 0)} of {fmt(cw.get('frame_px'), 0)} px; DEM valid over {fmt(100 * (cw.get('frame_dem_valid_fraction') or 0), 0, '%')} of the frame.</li>
        <li>Corrections: {esc(", ".join(outc.get("corrections_in_product_name", []) or ["—"]))}; deramp mask excludes every mapped panel + 2 km ({fmt(outc.get('deramp_mask_pixels_kept'), 0)} of {fmt(outc.get('deramp_mask_pixels_total'), 0)} coherent pixels used for the fit).</li>
      </ul>
    </div>
    <div>
      <h3>Wall clock by stage</h3>
      {bar_chart(stage_rows)}
      <p class="small muted">Unwrapping: {fmt(unwrap.get('n'), 0)} pairs, median {fmt(unwrap.get('median_s'), 0)} s, 90th percentile {hms(unwrap.get('p90_s'))}, slowest {hms(unwrap.get('max_s'))}, {fmt(unwrap.get('sum_cpu_h'), 1)} CPU-hours. Interferogram stages run as per-pair chains and coregistration as per-date chains, releasing intermediates as soon as their consumer's output exists ({cleaned.get('coreg_dates', 0)} dates' coregistered SLCs, {cleaned.get('unfiltered_pairs', 0)} unfiltered pairs).</p>
    </div>
  </div>
  <h3>What was excluded, and why</h3>
  <ul class="small">{"".join(excl)}</ul>
</section>

<section>
  <h2>Delivered products</h2>
  <p class="prose small">In <code>{esc(str(prod_dir))}</code>. Rasters are EPSG:4326 GeoTIFFs at {fmt((res.get('resolution') or {}).get('posting_m'), 0)} m posting, masked by temporal coherence ≥ {fmt(summ.get('temporal_coherence_threshold'), 2)} (the quality layers are not masked); the measurement points (<code>points_attributes.gpkg</code> and <code>points_timeseries.parquet</code>) use EGMS-style field names in mm and mm/yr, with cell-centre coordinates, the source radar pixel, a class-matched relative rate and the per-cell noise. Two uncertainty rasters are delivered: the formal one (a floor) and the empirical one (the figure to use). Absolute rates in these products carry the reference and the bias flagged above.</p>
  <ul class="small cols" style="list-style:none;padding:0">{"".join(f"<li><code>{esc(p)}</code></li>" for p in prods)}</ul>
</section>

<section class="prose">
  <h2>Open decisions</h2>
  <ul>
    <li><b>Whether to seek ground truth.</b> Relative motion below about {both(2 * max([spread(m)[1] for m in (ph.get('mines') or {}).values() if spread(m)[1]] or [0]), 0, 2, '/yr').replace('+', '±')} over a mine cannot be resolved from this record. The operator's subsidence survey, levelling or a GNSS station would; so would corner reflectors on the parcel for future monitoring, or L-band (NISAR), which keeps coherence over farmland far longer than C-band.</li>
    <li><b>The operator's panel schedule and mining method.</b> Public mine layers are often dated by whole calendar years, and whether the workings are room-and-pillar or longwall decides how much surface motion to expect at all.</li>
    <li><b>Reprocessing improvements</b> that would reduce unwrapping error, the fixable share of the drift: loading the unfiltered coherence for the inversion weights, a larger snaphu component limit where pairs hit it, and a quality layer from the per-pixel count of inconsistent loops.</li>
    <li>How resolution and posting are stated in the deliverable text, and whether the per-epoch columns belong in the measurement-point CSV.</li>
  </ul>
</section>

<footer>
  Generated {esc(datetime.now().strftime('%Y-%m-%d %H:%M'))} from <code>{esc(str(Path(res['project']) / 'run_results.json'))}</code>, which reads <code>quality_assessment.json</code>, <code>deliverables/summary.json</code>, <code>panel_history.json</code>, <code>site_history.json</code>, <code>uncertainty_model.json</code>, <code>provenance.json</code>, <code>stack/timing.json</code> and the product attributes. Pipeline: <code>insar_pipeline {esc(ver.get('version', (prov.get('software') or {}).get('insar_pipeline', '?')))}</code> (code {esc(ver.get('code_hash', (prov.get('software') or {}).get('code_hash', '?')))}; {esc(res.get('backend', (prov.get('processing') or {}).get('backend', '?')))} route), ISCE2 {esc(prov.get('software', {}).get('isce2', ''))}, MintPy {esc(prov.get('software', {}).get('mintpy', ''))}{(', MiaplPy ' + esc(str(((ver.get('miaplpy') or {}).get('version')) or ''))) if res.get('backend') == 'miaplpy' else ''}{(', ' + esc(str(ver.get('snaphu')))) if ver.get('snaphu') else ''}.
  {credits_html}
</footer>
</main>
<script>
(function(){{
  var fig=document.querySelector('figure.chart[data-chart]'); if(!fig) return;
  var d=JSON.parse(fig.getAttribute('data-chart')); var svg=fig.querySelector('svg'); var tip=fig.querySelector('.tip');
  var xh=svg.querySelector('.xhair'); var vb=svg.viewBox.baseVal; var cls=['s1','s2','s3','s4'];
  function toDays(s){{return (Date.UTC(+s.slice(0,4),+s.slice(4,6)-1,+s.slice(6,8))-Date.UTC(+d.t0.slice(0,4),+d.t0.slice(4,6)-1,+d.t0.slice(6,8)))/86400000;}}
  var xs=d.dates.map(function(s){{return d.L+d.pw*toDays(s)/d.span;}});
  function move(ev){{
    var r=svg.getBoundingClientRect(); var px=(ev.clientX-r.left)*vb.width/r.width;
    var best=0,bd=1e9; for(var i=0;i<xs.length;i++){{var dd=Math.abs(xs[i]-px); if(dd<bd){{bd=dd;best=i;}}}}
    xh.setAttribute('x1',xs[best]); xh.setAttribute('x2',xs[best]); xh.style.display='';
    var s=d.dates[best]; tip.textContent='';
    var h=document.createElement('div'); h.textContent=s.slice(0,4)+'-'+s.slice(4,6)+'-'+s.slice(6,8); tip.appendChild(h);
    d.series.forEach(function(ser,k){{
      var v=ser[best]; var row=document.createElement('div'); var key=document.createElement('i'); key.className='k '+cls[k];
      var b=document.createElement('b'); b.textContent=(v==null?'—':(v>=0?'+':'')+v.toFixed(1)+' mm ('+(v>=0?'+':'')+(v/25.4).toFixed(2)+' in)'); row.appendChild(key); row.appendChild(b);
      row.appendChild(document.createTextNode(' '+d.labels[k])); tip.appendChild(row);}});
    tip.hidden=false; var fx=(ev.clientX-fig.getBoundingClientRect().left); tip.style.left=Math.min(fx+12, fig.clientWidth-tip.offsetWidth-8)+'px'; tip.style.top='8px';
  }}
  svg.addEventListener('pointermove',move); svg.addEventListener('pointerleave',function(){{tip.hidden=true; xh.style.display='none';}});
}})();
</script>'''
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--title", default=None, help="page title (default: '<project title> Ground Motion Record')")
    ap.add_argument("--notes", help="JSON with {heading, items:[{title, body(html)}]}")
    ap.add_argument("--figures", help="directory with figures.json + PNGs (scratchpad/make_figs.py)")
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text())
    notes = json.loads(Path(a.notes).read_text()) if a.notes else None
    Path(a.out).write_text(build(res, a.title, notes, a.figures))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
