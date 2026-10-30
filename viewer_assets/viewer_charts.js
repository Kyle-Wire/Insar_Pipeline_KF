/* The results viewer's charts: pure functions from a spec to an SVG string,
 * no page. One definition of every chart, for the viewer now and - later -
 * for the report's figures (rendered headless by node), so the two cannot
 * disagree.
 *
 * Colours are written into the SVG (no CSS variables), so a downloaded
 * chart looks the same outside the page. theme('light') is what a download
 * uses; the page draws in its own theme. The categorical slots are the
 * dataviz reference palette's first four (validated: adjacent CVD and
 * normal-vision separation pass in both modes; slots 3-4 sit under 3:1 on
 * the light surface, so a chart with several series carries a legend and a
 * table view - never colour alone).
 *
 * Every chart returns {svg, model}: model is what the page's hover layer
 * needs (the plot box, x of each image, the series).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.InsarCharts = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const THEMES = {
    light: {surface: '#fcfcfb', ink: '#0b0b0b', ink2: '#52514e', muted: '#898781', grid: '#e1e0d9',
            axis: '#c3c2b7', band: 'rgba(120,128,140,0.20)', band2: 'rgba(120,128,140,0.34)',
            series: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'], down: '#b2182b', up: '#2166ac',
            draft: 'rgba(168,67,47,0.13)', window: 'rgba(42,120,214,0.07)'},
    dark: {surface: '#1a1a19', ink: '#f0efec', ink2: '#c3c2b7', muted: '#898781', grid: '#2c2c2a',
           axis: '#383835', band: 'rgba(170,176,186,0.18)', band2: 'rgba(170,176,186,0.30)',
           series: ['#3987e5', '#d95926', '#199e70', '#c98500'], down: '#ff8a8a', up: '#8ab8ff',
           draft: 'rgba(255,138,138,0.12)', window: 'rgba(57,135,229,0.10)'},
  };
  const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';

  function theme(mode) { return THEMES[mode] || THEMES.light; }

  function esc(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  }

  function signed(v, nd) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    const s = v.toFixed(nd);
    return Number(s) === 0 ? (0).toFixed(nd) : (v > 0 ? '+' : '') + s;
  }

  /* a round step giving at most ~5 gridlines over span */
  function niceStep(span, maxTicks = 5) {
    const raw = span / maxTicks, p = 10 ** Math.floor(Math.log10(raw || 1));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) return m * p;
    return 10 * p;
  }

  const dayOf = d => Date.UTC(+d.slice(0, 4), +d.slice(4, 6) - 1, +d.slice(6, 8));

  /* the stamp: every downloaded chart carries the report's own lines
   * (client_report.stamp_lines, sent by the server), the figure's kind and
   * window, and a DRAFT watermark until the report is final */
  /* lines wrapped at word boundaries to about `max` characters */
  function wrap(lines, max) {
    const out = [];
    for (const ln of lines) {
      let cur = '';
      for (const w of String(ln).split(' ')) {
        if (cur && (cur + ' ' + w).length > max) { out.push(cur); cur = w; } else cur = cur ? cur + ' ' + w : w;
      }
      if (cur) out.push(cur);
    }
    return out;
  }

  function stampSvg(spec, th, W, y) {
    let lines = (spec.stamp || []).slice();
    if (spec.kindText) lines.unshift(spec.kindText);
    lines = wrap(lines, Math.max(40, Math.floor((W - 16) / 5.2)));       // ~5.2 px a character at 9.5 px
    let s = '';
    lines.forEach((ln, k) => {
      s += `<text x="8" y="${y + 12 + k * 12}" font-size="9.5" fill="${th.muted}">${esc(ln)}</text>`;
    });
    return {svg: s, height: lines.length ? 8 + lines.length * 12 : 0};
  }

  function watermark(spec, th, W, H) {
    if (spec.final) return '';
    return `<text x="${W / 2}" y="${H / 2}" font-size="${Math.min(W, H) / 3.2}" font-weight="700" fill="${th.draft}"
      text-anchor="middle" dominant-baseline="middle" transform="rotate(-24 ${W / 2} ${H / 2})">DRAFT</text>`;
  }

  /* A time-series chart.
   * spec: {title, subtitle, unit ('in'|'mm'), dates [YYYYMMDD], series: [{name, y, color?, width?, dash?}],
   *        band: {lo, hi, label}?, fits: [{name, y0, y1, color?, dash?}]? (straight lines first -> last image),
   *        lines: [{name, y, color?, dash?}]? (other lines, e.g. the GNSS regional motion),
   *        stamp: [str], kindText, final, width?, height?, nd? (decimals on the axis) }
   * y values may be null (a missing image). */
  function seriesChart(spec, th = theme('light')) {
    const W = spec.width || 640, Hplot = spec.height || 250;
    const L = 56, R = 14, T = spec.subtitle ? 44 : 30, B = 30;
    // the legend's items, laid out in rows that fit the width (four long
    // names ran off the right edge on one row)
    const legendItems = [];
    // spec.background: {lines: [[y...]], label}: many thin grey lines behind
    // everything (a polygon's individual cells behind their average)
    const bg = spec.background && spec.background.lines && spec.background.lines.length ? spec.background : null;
    const bgColor = bg ? (bg.color || th.muted) : null;
    if (bg) legendItems.push({label: bg.label || 'individual series', color: bgColor});
    if (bg || spec.series.length > 1 || (spec.fits || []).length || (spec.lines || []).length || spec.band) {
      spec.series.forEach((sr, k) => legendItems.push({label: sr.name, color: sr.color || th.series[k % th.series.length], dash: sr.dash}));
      for (const f of spec.fits || []) legendItems.push({label: f.name, color: f.color || th.ink2, dash: f.dash || '6 3'});
      for (const l of spec.lines || []) legendItems.push({label: l.name, color: l.color || th.muted, dash: l.dash || '5 4'});
      if (spec.band) legendItems.push({label: spec.band.label || '95% range', color: th.band2, box: true});
    }
    let lrow = 0, lxp = L;
    for (const it of legendItems) {
      it.w = 30 + Math.min(W - L - R - 30, 6.1 * String(it.label).length);
      if (lxp > L && lxp + it.w > W - R) { lrow++; lxp = L; }
      it.x = lxp; it.row = lrow; lxp += it.w;
    }
    const legendRows = legendItems.length ? lrow + 1 : 0;
    const st = stampSvg(spec, th, W, 0);
    const H = T + Hplot + B + legendRows * 18 + st.height;
    const w = W - L - R, h = Hplot;
    const d = spec.dates, x0 = dayOf(d[0]), x1 = dayOf(d[d.length - 1]) || x0 + 1;
    const X = i => L + (dayOf(d[i]) - x0) / Math.max(1, x1 - x0) * w;
    const all = [];
    for (const s of spec.series) for (const v of s.y) if (v !== null && isFinite(v)) all.push(v);
    if (spec.band) for (const k of ['lo', 'hi']) for (const v of spec.band[k]) if (v !== null && isFinite(v)) all.push(v);
    for (const f of spec.fits || []) all.push(f.y0, f.y1);
    for (const l of spec.lines || []) for (const v of l.y) if (v !== null && isFinite(v)) all.push(v);
    if (bg) {                          // the grey lines widen the axis to their 1st-99th percentile, no further
      const pool = [];
      for (const ln of bg.lines) for (const v of ln) if (v !== null && isFinite(v)) pool.push(v);
      if (pool.length) {
        pool.sort((a, b) => a - b);
        const q = p => pool[Math.min(pool.length - 1, Math.max(0, Math.round(p * (pool.length - 1))))];
        all.push(q(0.01), q(0.99));
      }
    }
    if (all.length < 2) {
      return {svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} 60" width="${W}" height="60"><text x="12" y="34" font-family='${FONT}' font-size="13" fill="${th.muted}">Too few images to chart.</text></svg>`, model: null};
    }
    let lo = Math.min(0, ...all), hi = Math.max(0, ...all);
    const pad = Math.max((hi - lo) * 0.08, spec.unit === 'mm' ? 1 : 0.02);
    lo -= pad; hi += pad;
    const Y = v => T + (hi - v) / (hi - lo) * h;
    const step = niceStep(hi - lo), nd = spec.nd ?? (step < 0.1 ? 2 : step < 1 ? 1 : 0);
    let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family='${FONT}'>`;
    s += `<rect width="${W}" height="${H}" fill="${th.surface}"/>`;
    s += `<text x="${L}" y="18" font-size="13.5" font-weight="600" fill="${th.ink}">${esc(spec.title || '')}</text>`;
    if (spec.subtitle) s += `<text x="${L}" y="34" font-size="11.5" fill="${th.ink2}">${esc(spec.subtitle)}</text>`;
    // y grid and labels
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) {
      const y = Y(v).toFixed(1);
      s += `<line x1="${L}" x2="${L + w}" y1="${y}" y2="${y}" stroke="${th.grid}" stroke-width="1"/>`;
      s += `<text x="${L - 6}" y="${(+y + 3.5).toFixed(1)}" font-size="10.5" text-anchor="end" fill="${th.muted}">${signed(v, nd)}</text>`;
    }
    s += `<text x="12" y="${T + h / 2}" font-size="10.5" fill="${th.muted}" transform="rotate(-90 12 ${T + h / 2})" text-anchor="middle">${esc(spec.unit === 'mm' ? 'mm' : 'inches')}</text>`;
    s += `<line x1="${L}" x2="${L + w}" y1="${Y(0).toFixed(1)}" y2="${Y(0).toFixed(1)}" stroke="${th.axis}" stroke-width="1.2"/>`;
    // x ticks: years (or months for a short window)
    const yrs = (x1 - x0) / (365.25 * 864e5);
    const y0 = +d[0].slice(0, 4), yN = +d[d.length - 1].slice(0, 4);
    const ticks = [];
    if (yrs > 1.5) {
      const ys = yrs > 7 ? 2 : 1;
      for (let yr = y0 + 1; yr <= yN; yr += ys) ticks.push([Date.UTC(yr, 0, 1), String(yr)]);
      // a short record: mid-year ticks too, so the axis is never one label
      if (yrs <= 4) for (let yr = y0; yr <= yN; yr++) ticks.push([Date.UTC(yr, 6, 1), 'Jul']);
      for (let k = ticks.length - 1; k >= 0; k--) if (ticks[k][0] < x0 || ticks[k][0] > x1) ticks.splice(k, 1);
    } else {
      const M = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      const a = new Date(x0);
      for (let k = 1; k <= 24; k++) {
        const t = Date.UTC(a.getUTCFullYear(), a.getUTCMonth() + k, 1);
        if (t > x1) break;
        const dd = new Date(t);
        if (yrs > 0.8 && dd.getUTCMonth() % 2) continue;
        ticks.push([t, dd.getUTCMonth() === 0 ? String(dd.getUTCFullYear()) : M[dd.getUTCMonth()]]);
      }
    }
    for (const [tt, lab] of ticks) {
      const x = (L + (tt - x0) / Math.max(1, x1 - x0) * w).toFixed(1);
      s += `<line x1="${x}" x2="${x}" y1="${T + h}" y2="${T + h + 4}" stroke="${th.axis}"/>`;
      s += `<text x="${x}" y="${T + h + 16}" font-size="10.5" text-anchor="middle" fill="${th.muted}">${lab}</text>`;
    }
    // the background lines (clipped to the plot), the band, then fits and other lines, then the series
    if (bg) {
      const cid = `bgc${Math.random().toString(36).slice(2, 9)}`;
      s += `<clipPath id="${cid}"><rect x="${L}" y="${T}" width="${w}" height="${h}"/></clipPath><g clip-path="url(#${cid})" opacity="${bg.opacity ?? 0.35}">`;
      for (const ln of bg.lines) s += pathFor(ln, X, Y, bgColor, bg.width || 0.7, null);
      s += '</g>';
    }
    if (spec.band) {
      const up = [], dn = [];
      spec.band.hi.forEach((v, i) => { if (v !== null && isFinite(v)) up.push(`${X(i).toFixed(1)},${Y(v).toFixed(1)}`); });
      for (let i = spec.band.lo.length - 1; i >= 0; i--) {
        const v = spec.band.lo[i];
        if (v !== null && isFinite(v)) dn.push(`${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
      }
      s += `<polygon points="${up.concat(dn).join(' ')}" fill="${th.band}" stroke="none"/>`;
    }
    for (const l of spec.lines || []) s += pathFor(l.y, X, Y, l.color || th.muted, 1.2, l.dash || '5 4');
    for (const f of spec.fits || []) {
      s += `<line x1="${X(0).toFixed(1)}" x2="${X(d.length - 1).toFixed(1)}" y1="${Y(f.y0).toFixed(1)}" y2="${Y(f.y1).toFixed(1)}" stroke="${f.color || th.ink2}" stroke-width="1.4" stroke-dasharray="${f.dash || '6 3'}"/>`;
    }
    spec.series.forEach((sr, k) => {
      s += pathFor(sr.y, X, Y, sr.color || th.series[k % th.series.length], sr.width || 2, sr.dash || null);
    });
    s += `<rect x="${L}" y="${T}" width="${w}" height="${h}" fill="none" stroke="${th.grid}"/>`;
    // legend: line keys (a box for the band), in the rows laid out above
    let ly = T + h + 30;
    for (const it of legendItems) {
      const y = ly + it.row * 18, x = it.x;
      s += it.box ? `<rect x="${x}" y="${y - 8}" width="14" height="9" fill="${it.color}"/>`
                  : `<line x1="${x}" x2="${x + 16}" y1="${y - 4}" y2="${y - 4}" stroke="${it.color}" stroke-width="2" ${it.dash ? `stroke-dasharray="${it.dash}"` : ''}/>`;
      s += `<text x="${x + 21}" y="${y}" font-size="10.5" fill="${th.ink2}">${esc(it.label)}</text>`;
    }
    if (legendRows) ly += (legendRows - 1) * 18 + 10;
    const st2 = stampSvg(spec, th, W, ly);
    s += st2.svg + watermark(spec, th, W, H) + '</svg>';
    return {svg: s, model: {kind: 'series', L, T, w, h, W, H, xs: d.map((_, i) => X(i)), Y, lo, hi, dates: d,
                            series: spec.series, unit: spec.unit, nd: spec.unit === 'mm' ? 1 : 2}};
  }

  function pathFor(y, X, Y, color, width, dash) {
    let p = '', pen = false;
    y.forEach((v, i) => {
      if (v === null || !isFinite(v)) { pen = false; return; }
      p += `${pen ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`; pen = true;
    });
    return p ? `<path d="${p}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" ${dash ? `stroke-dasharray="${dash}"` : ''}/>` : '';
  }

  /* A histogram: bars for the selection, a line for the comparison
   * (farmland), both as a share of their own points so they compare.
   * spec: {title, subtitle, unitLabel, lo, hi, counts [n], compare: [n]?, compareLabel, label,
   *        marks: [{x, label}]?, stamp, kindText, final} */
  function histogramChart(spec, th = theme('light')) {
    const W = spec.width || 640, Hplot = spec.height || 210, L = 52, R = 14, T = spec.subtitle ? 44 : 30, B = 32;
    const st = stampSvg(spec, th, W, 0);
    const H = T + Hplot + B + 18 + st.height, w = W - L - R, h = Hplot;
    const n = spec.counts.length, tot = spec.counts.reduce((a, b) => a + b, 0) || 1;
    const share = spec.counts.map(c => 100 * c / tot);
    const cmpTot = spec.compare ? spec.compare.reduce((a, b) => a + b, 0) || 1 : 1;
    const cmp = spec.compare ? spec.compare.map(c => 100 * c / cmpTot) : null;
    const top = Math.max(1, ...share, ...(cmp || [0])) * 1.1;
    const X = v => L + (v - spec.lo) / (spec.hi - spec.lo) * w, Y = v => T + h - v / top * h;
    const bw = w / n;
    let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family='${FONT}'>`;
    s += `<rect width="${W}" height="${H}" fill="${th.surface}"/>`;
    s += `<text x="${L}" y="18" font-size="13.5" font-weight="600" fill="${th.ink}">${esc(spec.title || '')}</text>`;
    if (spec.subtitle) s += `<text x="${L}" y="34" font-size="11.5" fill="${th.ink2}">${esc(spec.subtitle)}</text>`;
    const ys = niceStep(top, 4);
    for (let v = 0; v <= top + 1e-9; v += ys) {
      s += `<line x1="${L}" x2="${L + w}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="${th.grid}"/>`;
      s += `<text x="${L - 6}" y="${(Y(v) + 3.5).toFixed(1)}" font-size="10.5" text-anchor="end" fill="${th.muted}">${v.toFixed(ys < 1 ? 1 : 0)}%</text>`;
    }
    share.forEach((v, i) => {
      if (!v) return;
      const x = L + i * bw + 1, y = Y(v);
      s += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1, bw - 2).toFixed(1)}" height="${(T + h - y).toFixed(1)}" fill="${th.series[0]}" rx="2"/>`;
    });
    if (cmp) {
      let p = `M${L},${Y(0).toFixed(1)}`;
      cmp.forEach((v, i) => { p += `L${(L + i * bw).toFixed(1)},${Y(v).toFixed(1)}L${(L + (i + 1) * bw).toFixed(1)},${Y(v).toFixed(1)}`; });
      s += `<path d="${p}L${L + w},${Y(0).toFixed(1)}" fill="none" stroke="${th.ink2}" stroke-width="1.6"/>`;
    }
    const xs = niceStep(spec.hi - spec.lo, 6), xnd = xs < 0.1 ? 2 : xs < 1 ? 1 : 0;
    for (let v = Math.ceil(spec.lo / xs) * xs; v <= spec.hi + 1e-9; v += xs) {
      s += `<line x1="${X(v).toFixed(1)}" x2="${X(v).toFixed(1)}" y1="${T + h}" y2="${T + h + 4}" stroke="${th.axis}"/>`;
      s += `<text x="${X(v).toFixed(1)}" y="${T + h + 16}" font-size="10.5" text-anchor="middle" fill="${th.muted}">${signed(v, xnd)}</text>`;
    }
    for (const mk of spec.marks || []) {
      if (mk.x < spec.lo || mk.x > spec.hi) continue;
      s += `<line x1="${X(mk.x).toFixed(1)}" x2="${X(mk.x).toFixed(1)}" y1="${T}" y2="${T + h}" stroke="${th.ink}" stroke-width="1.4" stroke-dasharray="3 3"/>`;
      s += `<text x="${(X(mk.x) + 4).toFixed(1)}" y="${T + 12}" font-size="10.5" fill="${th.ink}">${esc(mk.label)}</text>`;
    }
    s += `<text x="${L + w}" y="${T + h + 28}" font-size="10.5" text-anchor="end" fill="${th.muted}">${esc(spec.unitLabel || '')}</text>`;
    let lx = L, ly = T + h + 44;
    s += `<rect x="${lx}" y="${ly - 8}" width="14" height="9" fill="${th.series[0]}" rx="2"/><text x="${lx + 20}" y="${ly}" font-size="10.5" fill="${th.ink2}">${esc(spec.label || 'selection')}</text>`;
    lx += 30 + 6.1 * String(spec.label || 'selection').length;
    if (cmp) s += `<line x1="${lx}" x2="${lx + 16}" y1="${ly - 4}" y2="${ly - 4}" stroke="${th.ink2}" stroke-width="1.6"/><text x="${lx + 21}" y="${ly}" font-size="10.5" fill="${th.ink2}">${esc(spec.compareLabel || 'comparison')}</text>`;
    const st2 = stampSvg(spec, th, W, ly + 8);
    s += st2.svg + watermark(spec, th, W, H) + '</svg>';
    return {svg: s, model: {kind: 'bars', L, T, w, h, W, H, n, lo: spec.lo, hi: spec.hi, share, cmp,
                            counts: spec.counts, compare: spec.compare}};
  }

  /* Bars of a value per category (e.g. each calendar year's rate), zero
   * line, optional +/- error. spec: {title, subtitle, unitLabel, cats: [label], values: [v], errors?: [e],
   *  stamp, kindText, final, nd} */
  function barsChart(spec, th = theme('light')) {
    const W = spec.width || 640, Hplot = spec.height || 200, L = 56, R = 14, T = spec.subtitle ? 44 : 30, B = 30;
    const st = stampSvg(spec, th, W, 0);
    const H = T + Hplot + B + st.height + 6, w = W - L - R, h = Hplot;
    const vals = spec.values, errs = spec.errors || vals.map(() => 0);
    const all = vals.flatMap((v, i) => isFinite(v) ? [v - (errs[i] || 0), v + (errs[i] || 0)] : []);
    let lo = Math.min(0, ...all), hi = Math.max(0, ...all);
    const pad = Math.max((hi - lo) * 0.1, 0.01); lo -= pad; hi += pad;
    const Y = v => T + (hi - v) / (hi - lo) * h, n = vals.length, bw = w / Math.max(1, n);
    const nd = spec.nd ?? 2;
    let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family='${FONT}'>`;
    s += `<rect width="${W}" height="${H}" fill="${th.surface}"/>`;
    s += `<text x="${L}" y="18" font-size="13.5" font-weight="600" fill="${th.ink}">${esc(spec.title || '')}</text>`;
    if (spec.subtitle) s += `<text x="${L}" y="34" font-size="11.5" fill="${th.ink2}">${esc(spec.subtitle)}</text>`;
    const step = niceStep(hi - lo);
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) {
      s += `<line x1="${L}" x2="${L + w}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="${th.grid}"/>`;
      s += `<text x="${L - 6}" y="${(Y(v) + 3.5).toFixed(1)}" font-size="10.5" text-anchor="end" fill="${th.muted}">${signed(v, step < 0.1 ? 2 : 1)}</text>`;
    }
    s += `<line x1="${L}" x2="${L + w}" y1="${Y(0).toFixed(1)}" y2="${Y(0).toFixed(1)}" stroke="${th.axis}" stroke-width="1.2"/>`;
    vals.forEach((v, i) => {
      if (!isFinite(v)) return;
      const x = L + i * bw + bw * 0.2, bwi = bw * 0.6, y = Y(Math.max(0, v)), y2 = Y(Math.min(0, v));
      s += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bwi.toFixed(1)}" height="${Math.max(1, y2 - y).toFixed(1)}" fill="${v < 0 ? th.down : th.up}" rx="2"/>`;
      if (errs[i]) {
        const cx = x + bwi / 2;
        s += `<line x1="${cx.toFixed(1)}" x2="${cx.toFixed(1)}" y1="${Y(v + errs[i]).toFixed(1)}" y2="${Y(v - errs[i]).toFixed(1)}" stroke="${th.ink2}" stroke-width="1.2"/>`;
      }
      s += `<text x="${(x + bwi / 2).toFixed(1)}" y="${T + h + 16}" font-size="10.5" text-anchor="middle" fill="${th.muted}">${esc(spec.cats[i])}</text>`;
      // the value beyond the bar's end and its error bar, never on them
      const e = errs[i] || 0, ty = v < 0 ? Y(v - e) + 12 : Y(v + e) - 4;
      s += `<text x="${(x + bwi / 2).toFixed(1)}" y="${ty.toFixed(1)}" font-size="10" text-anchor="middle" fill="${th.ink2}">${signed(v, nd)}</text>`;
    });
    s += `<text x="12" y="${T + h / 2}" font-size="10.5" fill="${th.muted}" transform="rotate(-90 12 ${T + h / 2})" text-anchor="middle">${esc(spec.unitLabel || '')}</text>`;
    const st2 = stampSvg(spec, th, W, T + h + 24);
    s += st2.svg + watermark(spec, th, W, H) + '</svg>';
    return {svg: s, model: {kind: 'cats', L, T, w, h, W, H, n, cats: spec.cats, values: vals}};
  }

  return {theme, esc, seriesChart, histogramChart, barsChart, niceStep, THEMES};
}));
