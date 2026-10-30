/* node --test viewer_assets/tests/  (CI: .github/workflows/checks.yml)
 *
 * Small fixtures written here, so CI needs neither numpy nor a project. The
 * full parity with the Python reference (and so with results.analyse) is
 * tests/test_viewer_js_parity.py, run locally with the pinned environment. */
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const C = require(path.join(__dirname, '..', 'viewer_core.js'));
const K = require(path.join(__dirname, '..', 'viewer_charts.js'));

const dates = [];
for (let i = 0; i < 40; i++) {
  const d = new Date(Date.UTC(2020, 0, 5) + i * 12 * 864e5);
  dates.push(d.toISOString().slice(0, 10).replace(/-/g, ''));
}
const t = dates.map(d => (Date.UTC(+d.slice(0, 4), +d.slice(4, 6) - 1, +d.slice(6, 8)) - Date.UTC(2020, 0, 5)) / (365.25 * 864e5));

test('a window is the images inside it, both ends included', () => {
  assert.deepEqual(C.windowIndex(dates, '2020-01-05', '2020-02-10'), [0, 3]);
  assert.deepEqual(C.windowIndex(dates, '20200106', '20200210'), [1, 3]);
  assert.throws(() => C.windowIndex(dates, '2030-01-01', '2030-02-01'), /no images/);
});

test('a window needs 6 images over 3 months (results.MIN_DATES / MIN_YEARS)', () => {
  assert.match(C.windowProblem(t, 0, 4), /at least 6/);
  assert.match(C.windowProblem(t, 0, 6), /3 months/);           // 7 images, 72 days
  assert.equal(C.windowProblem(t, 0, 8), null);                 // 96 days
});

test('slopes: least squares over the finite values, NaN under 6', () => {
  const T = dates.length, Y = new Float32Array(2 * T);
  for (let i = 0; i < T; i++) { Y[i] = 3 + 2.5 * t[i]; Y[T + i] = i < 34 ? NaN : -4 * t[i]; }
  const b = C.slopes(Y, 2, T, t, 0, T - 1);
  assert.ok(Math.abs(b[0] - 2.5) < 1e-5);
  assert.ok(Math.abs(b[1] + 4) < 1e-5);                        // 6 finite values: measured
  const c = C.slopes(Y, 2, T, t, 0, 38);
  assert.ok(Number.isNaN(c[1]));                                // 5 finite values: not
});

test('the noise table: the larger of repeatability (nearest class) and spread (own class)', () => {
  const tab = C.nullTable({4: 1.0, 1: 3.0}, {4: 2.5, 2: 2.0});
  assert.deepEqual(tab, {0: 3, 1: 3, 2: 3, 3: 1, 4: 2.5});     // class 2 and 3: tie goes to the lower (1 for 2, ... )
  const per = C.perPointNull({1: 3, 4: 2.5}, Int8Array.from([-1, 0, 2, 3, 4]));
  assert.ok(Number.isNaN(per[0]));
  assert.deepEqual(Array.from(per.slice(1)), [3, 3, 2.5, 2.5]);
});

test('formula safety is areas.formula_safe / formula_safe_text', () => {
  assert.equal(C.formulaSafe('=cmd|x'), "'=cmd|x");
  assert.equal(C.formulaSafe('-1.5'), '-1.5');
  assert.equal(C.formulaSafe('\tTab'), "'\tTab");
  assert.equal(C.formulaSafeText('North,=HYPERLINK("x")'), 'North,\'=HYPERLINK("x")');
  assert.equal(C.formulaSafeText('a;-b'), "a;'-b");
  assert.equal(C.oneLine('a b\n  c'), 'a b c');
  assert.equal(C.csvRow(['=x', 'a,b', 1.5]), '\'=x,"a,b",1.5');
});

test('numbers never read -0.00, and stay beside a limit', () => {
  assert.equal(C.signed(-0.001, 2), '0.00');
  assert.equal(C.signed(0.126, 2), '+0.13');
  assert.equal(C.besideLimit(0.996, 1, 2), '0.99');
  assert.equal(C.besideLimit(1.004, 1, 2), '1.00');
});

test('a polygon holds the points inside its outline and outside its holes', () => {
  const lon = Float64Array.from([0.5, 1.5, 0.5, 0.2]), lat = Float64Array.from([0.5, 0.5, 0.5, 0.2]);
  const group = Uint8Array.from([0, 0, 1, 0]);
  const outer = [[0, 0], [1, 0], [1, 1], [0, 1]], hole = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]];
  assert.deepEqual(C.inside(lon, lat, group, [outer]), [0, 3]);           // group 1 (a feature's sample) left out
  assert.deepEqual(C.inside(lon, lat, group, [outer, hole]), [0]);
});

test('the bundle decodes: header, then 8-aligned arrays', () => {
  const header = JSON.stringify({format: 1, meta: {n: 2}, arrays: [
    {name: 'a', dtype: '<f8', shape: [2], offset: 0, nbytes: 16}, {name: 'b', dtype: 'u1', shape: [2], offset: 16, nbytes: 2}]});
  let hb = Buffer.from(header);
  hb = Buffer.concat([hb, Buffer.alloc((8 - ((4 + hb.length) % 8)) % 8, 32)]);
  const buf = Buffer.alloc(4 + hb.length + 24);
  buf.writeUInt32LE(hb.length, 0); hb.copy(buf, 4);
  buf.writeDoubleLE(1.25, 4 + hb.length); buf.writeDoubleLE(-2, 4 + hb.length + 8);
  buf[4 + hb.length + 16] = 7; buf[4 + hb.length + 17] = 9;
  const B = C.decodeBundle(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
  assert.deepEqual(Array.from(B.arrays.a), [1.25, -2]);
  assert.deepEqual(Array.from(B.arrays.b), [7, 9]);
});

test('charts are SVG, escape what they are given, and say DRAFT until final', () => {
  const y = dates.map((_, i) => -0.01 * i);
  const r = K.seriesChart({title: '<b>x</b>', unit: 'in', dates, series: [{name: 'a&b', y}, {name: 'c', y}], stamp: ['line'], final: false});
  assert.match(r.svg, /^<svg xmlns="http:\/\/www.w3.org\/2000\/svg"/);
  assert.ok(r.svg.includes('&lt;b&gt;x&lt;/b&gt;') && r.svg.includes('a&amp;b') && r.svg.includes('>DRAFT<'));
  assert.ok(!K.seriesChart({title: 't', unit: 'in', dates, series: [{name: 'a', y}], final: true}).svg.includes('>DRAFT<'));
  assert.equal(r.model.xs.length, dates.length);
});

test('a long legend wraps onto rows inside the chart', () => {
  const y = dates.map((_, i) => -0.01 * i);
  const names = [1, 2, 3, 4].map(k => `${k}: a polygon with a long name (511 points): -1.42 in/yr`);
  const r = K.seriesChart({title: 't', unit: 'in', dates, series: names.map(name => ({name, y})), width: 560, final: true});
  const xs = [...r.svg.matchAll(/<text x="([\d.]+)" y="([\d.]+)" font-size="10.5" fill="[^"]+">(\d: a polygon[^<]*)<\/text>/g)];
  assert.equal(xs.length, 4);
  for (const m of xs) assert.ok(+m[1] + 6.1 * m[3].length <= 560 + 1, m[3]);
  assert.ok(new Set(xs.map(m => m[2])).size > 1, 'more than one row');
});
