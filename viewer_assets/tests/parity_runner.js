/* Run viewer_core on a bundle for tests/test_viewer_js_parity.py:
 *   node parity_runner.js <bundle.bin> <request.json>
 * prints one JSON object (NaN as null). */
'use strict';
const fs = require('fs');
const path = require('path');
const C = require(path.join(__dirname, '..', 'viewer_core.js'));

const [binPath, reqPath] = process.argv.slice(2);
const raw = fs.readFileSync(binPath);
const B = C.decodeBundle(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const req = JSON.parse(fs.readFileSync(reqPath, 'utf8'));
const list = a => a ? Array.from(a, v => (v === null || !isFinite(v) ? null : v)) : null;
const out = {windows: [], polygons: [], series: [], check: C.wholeRecordCheck(B), safe: {}};
for (const w of req.windows || []) {
  const W = C.windowFigures(B, w.i0, w.i1, w.table || null);
  out.windows.push({kind: W.kind, table: W.table, spread: W.spread || null, n_sample: W.n_sample || null,
                    los: list(W.los), vertical: list(W.vertical), snr: list(W.snr),
                    signal_to_threshold: list(W.signal_to_threshold), sd_vertical: list(W.sd_vertical),
                    change_los: list(W.change_los), change_los_ends: list(W.change_los_ends)});
}
for (const p of req.polygons || []) {
  const W = C.windowFigures(B, p.i0, p.i1, null);
  const idx = C.inside(B.arrays.lon, B.arrays.lat, B.arrays.group, p.rings);
  const st = C.polygonStats(B, W, idx);
  delete st.rows;
  out.polygons.push({idx, stats: st});
}
for (const s of req.series || []) out.series.push(C.pointSeries(B, s.j, s.i0, s.i1, s.vertical));
for (const t of req.safe || []) out.safe[t] = {cell: C.formulaSafe(t), text: C.formulaSafeText(t), line: C.oneLine(t)};
out.fmt = (req.fmt || []).map(([v, nd, lim]) => ({signed: C.signed(v, nd), beside: lim === null ? null : C.besideLimit(v, lim, nd)}));
process.stdout.write(JSON.stringify(out));
