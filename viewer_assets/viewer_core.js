/* The results viewer's arithmetic: pure functions over typed arrays, no page.
 *
 * Every figure here is viewer_bundle.py's (window_figures and friends), and
 * tests/test_viewer_js_parity.py runs this file under node against it. Keep
 * the two in step: a change here without the same change there fails that
 * test. Over the whole record the page shows the STORED figures (the
 * report's, the KMZ's, the rasters'); a window is computed here.
 *
 * A classic script, not an ES module: the client copy opens from file://,
 * where module scripts are refused. In a page it defines window.InsarCore;
 * under node, module.exports.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.InsarCore = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const MIN_DATES = 6;          // results.MIN_DATES
  const MIN_YEARS = 0.25;       // results.MIN_YEARS
  const END_DAYS = 30;          // results.analyse ends()
  const Z95 = 1.96;             // the per-point threshold of the report, the rasters and the KMZ
  const MAD_SD = 1.4826;        // significance.farmland_cell_spread
  const N_CLASSES = 5;          // significance.CLASS_EDGES
  const MM_IN = 25.4;

  // ------------------------------------------------------------ the bundle
  const DTYPES = {
    '<f8': Float64Array, '<f4': Float32Array, '|u1': Uint8Array, '|i1': Int8Array, 'u1': Uint8Array,
    'i1': Int8Array, '<i4': Int32Array, '<u4': Uint32Array, '<i2': Int16Array, '<u2': Uint16Array,
  };

  /* viewer_bundle.encode's binary: a 4-byte little-endian header length,
   * the JSON header, then each array at its 8-aligned offset. */
  function decodeBundle(buf) {
    const dv = new DataView(buf);
    const n = dv.getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
    const base = 4 + n, arrays = {}, shapes = {};
    for (const s of header.arrays) {
      const T = DTYPES[s.dtype];
      if (!T) throw new Error(`unknown array type ${s.dtype} for ${s.name}`);
      const count = s.shape.reduce((a, b) => a * b, 1);
      // a fresh copy: the offset of a view must be a multiple of its element size
      arrays[s.name] = new T(buf.slice(base + s.offset, base + s.offset + s.nbytes), 0, count);
      shapes[s.name] = s.shape;
    }
    return {header, meta: header.meta, arrays, shapes};
  }

  // ------------------------------------------------------------ windows
  const ymd = s => String(s || '').replace(/-/g, '').slice(0, 8);

  /* [first, last] image index inside [start, end], inclusive (results.analyse). */
  function windowIndex(dates, start, end) {
    const a = start ? ymd(start) : dates[0], b = end ? ymd(end) : dates[dates.length - 1];
    let i0 = -1, i1 = -1;
    for (let i = 0; i < dates.length; i++) {
      if (dates[i] >= a && dates[i] <= b) { if (i0 < 0) i0 = i; i1 = i; }
    }
    if (i0 < 0) throw new Error(`no images between ${a} and ${b}`);
    return [i0, i1];
  }

  function windowProblem(t, i0, i1) {
    const n = i1 - i0 + 1;
    if (n < MIN_DATES) return `only ${n} images in the window; at least ${MIN_DATES} are needed`;
    const span = t[i1] - t[i0];
    if (span < MIN_YEARS) return `the window spans ${(span * 12).toFixed(1)} months of images; at least 3 months are needed`;
    return null;
  }

  function dayNumbers(dates) {
    const d = s => Date.UTC(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8)) / 864e5;
    const d0 = d(dates[0]);
    return dates.map(s => Math.round(d(s) - d0));
  }

  // ------------------------------------------------------------ fits
  /* Each row's least-squares slope over its finite values in columns
   * i0..i1 (row-major Y, T columns), NaN with fewer than MIN_DATES -
   * field_points.slopes / results._fit_rates. */
  function slopes(Y, N, T, t, i0, i1, rows, skip) {
    const R = rows ? rows.length : N, out = new Float64Array(R);
    for (let r = 0; r < R; r++) {
      const base = (rows ? rows[r] : r) * T;
      let n = 0, st = 0, sy = 0, stt = 0, sty = 0;
      for (let i = i0; i <= i1; i++) {
        if (skip && skip[i]) continue;                           // a date the delivered rate leaves out
        const y = Y[base + i];
        if (y !== y) continue;                                   // NaN
        const x = t[i];
        n++; st += x; sy += y; stt += x * x; sty += x * y;
      }
      const den = n * stt - st * st;
      out[r] = n < MIN_DATES || !(den > 0) ? NaN : (n * sty - st * sy) / den;
    }
    return out;
  }

  /* Each row's fit (slope and intercept) over i0..i1. */
  function fitRow(Y, T, row, t, i0, i1) {
    let n = 0, st = 0, sy = 0, stt = 0, sty = 0;
    for (let i = i0; i <= i1; i++) {
      const y = Y[row * T + i];
      if (y !== y) continue;
      const x = t[i]; n++; st += x; sy += y; stt += x * x; sty += x * y;
    }
    const den = n * stt - st * st;
    if (n < 3 || !(den > 0)) return null;
    const b = (n * sty - st * sy) / den;
    return {slope: b, intercept: (sy - b * st) / n, n};
  }

  /* The axis MintPy fitted the delivered rates on (field_points.fit_axis):
   * its decimal years, skipping the dates it left out of the velocity fit
   * (exclude_date.txt); a table without it: t, every date. Only the load
   * check uses it - a window's figures are results.analyse's, on t over
   * every date. */
  function fitAxis(m) {
    if (!m._fit) {
      const T = m.dates.length, skip = new Uint8Array(T), ex = new Set(m.fit_exclude || []);
      for (let i = 0; i < T; i++) if (ex.has(m.dates[i])) skip[i] = 1;
      m._fit = {t: Float64Array.from(m.t_fit || m.t), skip};
    }
    return m._fit;
  }

  function median(a) {
    const x = Float64Array.from(a).sort();
    const n = x.length;
    if (!n) return NaN;
    return n % 2 ? x[(n - 1) / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]);
  }

  /* numpy.percentile's default (linear) */
  function percentile(a, p) {
    const x = Float64Array.from(a).sort();
    const n = x.length;
    if (!n) return NaN;
    const h = (n - 1) * p / 100, lo = Math.floor(h), hi = Math.min(n - 1, lo + 1);
    return x[lo] + (h - lo) * (x[hi] - x[lo]);
  }

  function madSd(a) {
    const m = median(a);
    return MAD_SD * median(Array.from(a, v => Math.abs(v - m)));
  }

  // ------------------------------------------------------------ the noise
  function nearest(table, i) {
    if (Object.prototype.hasOwnProperty.call(table, i)) return table[i];
    let best = null, bd = Infinity;
    for (const k of Object.keys(table).map(Number).sort((a, b) => a - b)) {
      const d = Math.abs(k - i);
      if (d < bd) { bd = d; best = k; }                          // ties: the lower (sorted, strict <)
    }
    return best === null ? null : table[best];
  }

  function cleanTable(t) {
    const out = {};
    for (const [k, v] of Object.entries(t || {})) if (v !== null && v !== undefined && isFinite(v)) out[+k] = +v;
    return out;
  }

  /* significance.cell_null_by_class */
  function nullTable(emp, spread) {
    emp = cleanTable(emp); spread = cleanTable(spread);
    const out = {};
    for (let i = 0; i < N_CLASSES; i++) {
      const vals = [nearest(emp, i), Object.prototype.hasOwnProperty.call(spread, i) ? spread[i] : null]
        .filter(v => v !== null && isFinite(v));
      if (vals.length) out[i] = Math.round(Math.max(...vals) * 1000) / 1000;
    }
    return out;
  }

  /* client_report.cell_sigma_map on the points */
  function perPointNull(table, kcls) {
    table = cleanTable(table);
    const out = new Float64Array(kcls.length).fill(NaN);
    if (!Object.keys(table).length) return out;
    const v = [];
    for (let i = 0; i < N_CLASSES; i++) v[i] = nearest(table, i);
    for (let j = 0; j < kcls.length; j++) { const k = kcls[j]; if (k >= 0 && k < N_CLASSES) out[j] = v[k]; }
    return out;
  }

  // ------------------------------------------------------------ the figures
  /* viewer_bundle.window_figures. B: decodeBundle's result. */
  function windowFigures(B, i0, i1, table) {
    const m = B.meta, a = B.arrays, t = m.t, dates = m.dates, T = dates.length, N = m.n;
    const whole = i0 === 0 && i1 === T - 1;
    // the whole record is the stored figures, whatever its length
    const why = whole ? null : windowProblem(t, i0, i1);
    if (why) throw new Error(why);
    const anchored = !!m.anchored, bgV = anchored ? (m.bg_v || 0) : 0, bgU = anchored ? m.bg_u : null;
    const up = a.up, C = a.cell_los;
    const out = {i0, i1, first: dates[i0], last: dates[i1], n_images: i1 - i0 + 1, years: t[i1] - t[i0]};
    if (C) {
      const ch = new Float64Array(N), ends = new Float64Array(N);
      const d = dayNumbers(dates);
      for (let j = 0; j < N; j++) {
        const b = j * T;
        ch[j] = C[b + i1] - C[b + i0];
        let s0 = 0, n0 = 0, s1 = 0, n1 = 0;
        for (let i = i0; i <= i1; i++) {
          const y = C[b + i];
          if (y !== y) continue;
          if (d[i] - d[i0] <= END_DAYS) { s0 += y; n0++; }
          if (d[i1] - d[i] <= END_DAYS) { s1 += y; n1++; }
        }
        ends[j] = n0 && n1 ? s1 / n1 - s0 / n0 : NaN;
      }
      out.change_los = ch; out.change_los_ends = ends;
    }
    let los, nul;
    if (whole) {
      out.kind = 'report';
      los = Float64Array.from(a.los_rel);
      nul = Float64Array.from(a.sd_v, (v, j) => isFinite(v) ? v * up[j] : NaN);
      out.table = cleanTable(m.rep_cell_null_by_class);
      out.change_los = Float64Array.from(a.change_los);
    } else {
      if (!C) throw new Error('these points were built before date windows: rebuild the client KMZ');
      los = slopes(C, N, T, t, i0, i1);
      let tab;
      if (table) { out.kind = 'tested'; tab = cleanTable(table); }
      else {
        out.kind = 'look';
        const S = farmlandSample(B);
        const rf = new Float64Array(S.n);
        const r1 = slopes(C, N, T, t, i0, i1, S.pointRows);
        rf.set(r1, 0);
        if (S.extra) rf.set(slopes(a.far_los, S.extra, T, t, i0, i1), r1.length);
        const trueN = cleanTable(m.far_cells_by_class), minN = m.min_spread_cells || 200;
        const spread = {}, nS = {};
        for (let k = 0; k < N_CLASSES; k++) {
          const x = [];
          for (let r = 0; r < S.n; r++) if (S.cls[r] === k && isFinite(rf[r])) x.push(rf[r]);
          nS[k] = x.length;
          if ((trueN[k] || 0) >= minN && x.length) spread[k] = Math.round(madSd(x) * 1000) / 1000;
        }
        tab = nullTable(m.emp_by_class, spread);
        out.spread = spread; out.n_sample = nS;
      }
      out.table = tab;
      nul = perPointNull(tab, a.kcls);
    }
    const vrel = new Float64Array(N), vert = new Float64Array(N), sd = new Float64Array(N);
    const snr = new Float64Array(N), stt = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      vrel[j] = los[j] / up[j]; vert[j] = vrel[j] + bgV;
      const ok = isFinite(nul[j]) && nul[j] > 0;
      sd[j] = ok ? Math.hypot(nul[j] / up[j], bgU || 0) : NaN;
      snr[j] = ok && isFinite(los[j]) ? Math.abs(los[j]) / nul[j] : NaN;
      stt[j] = ok && isFinite(los[j]) ? Math.abs(los[j]) / (Z95 * nul[j]) : NaN;
    }
    Object.assign(out, {los, vertical_rel: vrel, vertical: vert, null: nul, sd_vertical: sd, snr,
                        signal_to_threshold: stt});
    // a combined project (combine_viewer): the line-of-sight slot carries
    // east-west, which the page labels as such (indicative)
    if (m.combined && a.east) {
      out.los = Float64Array.from(a.east);
      if (a.east_change) out.change_los = Float64Array.from(a.east_change);
    }
    return out;
  }

  /* the farmland cells an untested window's noise is measured on:
   * field points that are the report's farmland, then the stored top-up */
  function farmlandSample(B) {
    const a = B.arrays, N = B.meta.n;
    const rows = [];
    for (let j = 0; j < N; j++) if (a.group[j] === 0 && a.far[j]) rows.push(j);
    const extra = a.far_cls ? a.far_cls.length : 0;
    const cls = new Int8Array(rows.length + extra);
    rows.forEach((j, r) => { cls[r] = a.kcls[j]; });
    if (extra) cls.set(a.far_cls, rows.length);
    return {pointRows: Int32Array.from(rows), extra, cls, n: rows.length + extra};
  }

  /* the viewer's load check: over the whole record each point's own-cell
   * series reproduces the stored rate and change */
  function wholeRecordCheck(B) {
    const m = B.meta, a = B.arrays, T = m.dates.length, N = m.n;
    if (!a.cell_los) return {rate: null, change: null};
    const F = fitAxis(m);
    const b = slopes(a.cell_los, N, T, F.t, 0, T - 1, null, F.skip);
    let e1 = -1, e2 = -1;
    for (let j = 0; j < N; j++) {
      const d1 = Math.abs(b[j] - a.los_rel[j]);
      if (isFinite(d1) && d1 > e1) e1 = d1;
      const d2 = Math.abs(a.cell_los[j * T + T - 1] - a.cell_los[j * T] - a.change_los[j]);
      if (isFinite(d2) && d2 > e2) e2 = d2;
    }
    return {rate: e1 < 0 ? null : e1, change: e2 < 0 ? null : e2};
  }

  // ------------------------------------------------------------ series for a chart
  /* A point's series over the window, zeroed on the window's own fit at the
   * window's first image (so the fitted line starts at 0 and ends at
   * rate x years): vertical = LOS / up + the regional GNSS rate x time
   * when anchored (field_points.series_vertical_in, on the window). mm. */
  function pointSeries(B, j, i0, i1, vertical) {
    const m = B.meta, a = B.arrays, T = m.dates.length, t = m.t, C = a.cell_los;
    const up = a.up[j], bg = vertical && m.anchored ? (m.bg_v || 0) : 0;
    const f = fitRow(C, T, j, t, i0, i1);
    const y = [];
    const zero = f ? f.intercept + f.slope * t[i0] : 0;
    for (let i = i0; i <= i1; i++) {
      const v = C[j * T + i];
      y.push(v !== v ? null : ((v - zero) / (vertical ? up : 1)) + bg * (t[i] - t[i0]));
    }
    return {y, fit: f ? {rate: f.slope / (vertical ? up : 1) + bg} : null};
  }

  /* The mean series of many points (rows), each zeroed as pointSeries. */
  function meanSeries(B, rows, i0, i1, vertical) {
    const n = i1 - i0 + 1, s = new Float64Array(n), c = new Int32Array(n);
    for (const j of rows) {
      const p = pointSeries(B, j, i0, i1, vertical).y;
      for (let i = 0; i < n; i++) if (p[i] !== null && isFinite(p[i])) { s[i] += p[i]; c[i]++; }
    }
    return Array.from(s, (v, i) => c[i] ? v / c[i] : null);
  }

  // ------------------------------------------------------------ polygons
  function inRing(x, y, ring) {
    let inside = false;
    for (let i = 0, k = ring.length - 1; i < ring.length; k = i++) {
      const xi = ring[i][0], yi = ring[i][1], xk = ring[k][0], yk = ring[k][1];
      if (((yi > y) !== (yk > y)) && (x < (xk - xi) * (y - yi) / (yk - yi) + xi)) inside = !inside;
    }
    return inside;
  }

  /* field points (group 0) inside rings [[lon, lat], ...]: the first the
   * outline, the rest holes (results_viewer._inside) */
  function inside(lon, lat, group, rings) {
    if (!rings || !rings.length || rings[0].length < 3) throw new Error('draw a polygon with at least three corners');
    let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
    for (const [x, y] of rings[0]) { w = Math.min(w, x); e = Math.max(e, x); s = Math.min(s, y); n = Math.max(n, y); }
    const out = [];
    for (let j = 0; j < lon.length; j++) {
      if (group[j] !== 0) continue;
      const x = lon[j], y = lat[j];
      if (x < w || x > e || y < s || y > n || !inRing(x, y, rings[0])) continue;
      let hole = false;
      for (let h = 1; h < rings.length; h++) if (inRing(x, y, rings[h])) { hole = true; break; }
      if (!hole) out.push(j);
    }
    return out;
  }

  /* the plain average of the points inside (results_viewer._poly_stats,
   * on the window's figures W) */
  function polygonStats(B, W, idx) {
    const a = B.arrays, m = B.meta;
    const ok = idx.filter(j => !a.fast[j]);
    const out = {n: idx.length, n_too_fast: idx.length - ok.length};
    if (!ok.length) return out;
    const fin = arr => ok.map(j => arr[j]).filter(v => isFinite(v));
    const mean = arr => { const x = fin(arr); return x.length ? x.reduce((p, q) => p + q, 0) / x.length : null; };
    const vin = fin(W.vertical).map(v => v / MM_IN);
    const snr = fin(W.snr);
    Object.assign(out, {
      vertical_in_yr: vin.length ? vin.reduce((p, q) => p + q, 0) / vin.length : null,
      vertical_in_yr_p10: vin.length ? percentile(vin, 10) : null,
      vertical_in_yr_p90: vin.length ? percentile(vin, 90) : null,
      relative_in_yr: (mean(W.vertical_rel) ?? NaN) / MM_IN,
      los_in_yr: (mean(W.los) ?? NaN) / MM_IN,
      los_change_in: W.change_los ? (mean(W.change_los) ?? NaN) / MM_IN : null,
      median_snr: snr.length ? median(snr) : null,
      percent_standing_out: snr.length ? 100 * snr.filter(v => v >= Z95).length / snr.length : null,
      percent_by_chance: m.snr_chance_pct, mean_coherence: mean(a.tc), anchored: !!m.anchored, rows: ok,
    });
    return out;
  }

  // ------------------------------------------------------------ distributions
  function histogram(values, lo, hi, nbins) {
    const c = new Array(nbins).fill(0), w = (hi - lo) / nbins;
    let below = 0, above = 0;
    for (const v of values) {
      if (!isFinite(v)) continue;
      if (v < lo) below++; else if (v >= hi) { if (v === hi) c[nbins - 1]++; else above++; }
      else c[Math.min(nbins - 1, Math.floor((v - lo) / w))]++;
    }
    return {counts: c, lo, hi, width: w, below, above};
  }

  /* each calendar year of the window with >= MIN_DATES images over >=
   * MIN_YEARS: the slope of a mean series (results' annual rule) */
  function yearlyRates(dates, t, y, i0) {
    const by = new Map();
    y.forEach((v, k) => {
      const yr = dates[i0 + k].slice(0, 4);
      if (!by.has(yr)) by.set(yr, []);
      if (v !== null && isFinite(v)) by.get(yr).push([t[i0 + k], v]);
    });
    const out = [];
    for (const [yr, pts] of by) {
      if (pts.length < MIN_DATES || pts[pts.length - 1][0] - pts[0][0] < MIN_YEARS) continue;
      let n = 0, st = 0, sy = 0, stt = 0, sty = 0;
      for (const [x, v] of pts) { n++; st += x; sy += v; stt += x * x; sty += x * v; }
      out.push({year: +yr, n_images: pts.length, rate: (n * sty - st * sy) / (n * stt - st * st)});
    }
    return out;
  }

  // ------------------------------------------------------------ words and numbers
  /* report_insights.signed / no negative zero: '+0.12', '-0.40', '0.00' */
  function signed(v, nd = 2) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    const s = v.toFixed(nd);
    if (Number(s) === 0) return (0).toFixed(nd);
    return (v > 0 ? '+' : '') + s;
  }

  function fixed(v, nd = 2) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    const s = v.toFixed(nd);
    return Number(s) === 0 ? (0).toFixed(nd) : s;
  }

  /* client_kmz._beside_limit: never shown at or above a limit it is below */
  function besideLimit(x, limit, nd = 2) {
    let s = x.toFixed(nd);
    if (x < limit && Number(s) >= limit) s = (limit - 10 ** -nd).toFixed(nd);
    return s;
  }

  function fmtDate(d) {
    const m = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return `${+d.slice(6, 8)} ${m[+d.slice(4, 6) - 1]} ${d.slice(0, 4)}`;
  }
  const isoDate = d => `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`;

  // areas.one_line / formula_safe / formula_safe_text (SEC-05)
  const CONTROL = new RegExp('[\\x00-\\x1f\\x7f\\u2028\\u2029]', 'g');
  const FORMULA_START = ['=', '+', '-', '@'];
  const CELL_FORMULA_START = FORMULA_START.concat(['\t', '\r']);
  const PLAIN_NUMBER = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;

  function oneLine(text) {
    return String(text ?? '').replace(CONTROL, ' ').replace(/\s+/g, ' ').trim();
  }

  function formulaSafe(value) {
    if (typeof value !== 'string') return value;
    if (CELL_FORMULA_START.some(c => value.startsWith(c)) && !PLAIN_NUMBER.test(value)) return "'" + value;
    return value;
  }

  function formulaSafeText(text) {
    return oneLine(text).split(/([,;])/).map(piece => {
      const body = piece.replace(/^[ "]+/, '');
      if (FORMULA_START.some(c => body.startsWith(c)) && !PLAIN_NUMBER.test(body.replace(/[ "]+$/, ''))) {
        return piece.slice(0, piece.length - body.length) + "'" + body;
      }
      return piece;
    }).join('');
  }

  /* one CSV row (RFC 4180 quoting) of cells already made safe */
  function csvRow(cells) {
    return cells.map(c => {
      const s = c === null || c === undefined ? '' : String(formulaSafe(typeof c === 'number' ? String(c) : c));
      return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    }).join(',');
  }

  /* '<project>_<what>_<start>_<end>' for a downloaded file */
  function fileName(project, what, first, last, ext) {
    const slug = s => String(s || '').replace(/[^A-Za-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'results';
    return `${slug(project)}_${slug(what)}_${first}_${last}.${ext}`;
  }

  return {
    MIN_DATES, MIN_YEARS, END_DAYS, Z95, MM_IN, N_CLASSES,
    decodeBundle, windowIndex, windowProblem, dayNumbers, slopes, fitRow, median, percentile, madSd,
    nearest, nullTable, perPointNull, windowFigures, farmlandSample, wholeRecordCheck, pointSeries, meanSeries,
    inRing, inside, polygonStats, histogram, yearlyRates, signed, fixed, besideLimit, fmtDate, isoDate,
    oneLine, formulaSafe, formulaSafeText, csvRow, fileName,
  };
}));
