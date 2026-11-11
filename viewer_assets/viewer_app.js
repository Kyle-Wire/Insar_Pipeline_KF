/* The results viewer page: the map, the date window, the panels, the
 * charts and the downloads. The arithmetic is viewer_core.js's (held to the
 * Python reference by the tests) and the charts viewer_charts.js's; this
 * file only wires them to the page.
 *
 * Live: the page is served by results_viewer.py and reads /api/bundle.
 * Static (the client's offline copy, phase 3): window.INSAR_BUNDLE holds
 * the data and there is no server - the server-only actions hide.
 */
(function () {
  'use strict';
  const C = window.InsarCore, K = window.InsarCharts;
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const STATIC = !!window.INSAR_BUNDLE_STATIC;
  const MM_IN = C.MM_IN;
  // Inside the app's results section (11 Nov 2026): the app's theme comes in the
  // link (#...&theme=dark) and again by message from the app's own page only.
  const APP_ORIGINS = ['http://localhost:8767', 'http://127.0.0.1:8767'];
  {
    const m = (location.hash || '').match(/[#&]theme=(dark|light)\b/);
    if (m) document.documentElement.setAttribute('data-theme', m[1]);
  }
  window.addEventListener('message', e => {
    if (!APP_ORIGINS.includes(e.origin) || !e.data || !['dark', 'light'].includes(e.data.theme)) return;
    document.documentElement.setAttribute('data-theme', e.data.theme);
    if (typeof points !== 'undefined') points.redraw();
  });

  function toast(msg, bad = false, ms = 4500) {
    const t = $('toast'); t.textContent = msg; t.className = bad ? 'bad' : ''; t.hidden = false;
    clearTimeout(toast.t); toast.t = setTimeout(() => { t.hidden = true; }, bad ? ms * 2 : ms);
  }

  // ------------------------------------------------------------ the server
  const TOKEN = (document.querySelector('meta[name="insar-token"]') || {}).content || '';
  async function post(path, body) {
    try {
      return await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Insar-Token': TOKEN},
                                body: JSON.stringify(body || {})});
    } catch (e) {
      throw new Error('The viewer server is not running. Start it again with "insar viewer" (or m in the dashboard).');
    }
  }
  async function api(path, body) {
    const r = await post(path, body);
    let d = {};
    try { d = await r.json(); } catch (e) { /* not JSON */ }
    if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
    return d;
  }
  async function apiBinary(path, body) {
    const r = await post(path, body);
    if (!r.ok) {
      let d = {};
      try { d = await r.json(); } catch (e) { /* not JSON */ }
      throw new Error(d.error || `HTTP ${r.status}`);
    }
    return r.arrayBuffer();
  }
  function b64bytes(b64) {
    const s = atob(b64), u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return u;
  }
  function download(name, blob) {
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1500);
  }
  const darkMode = () => {
    const forced = document.documentElement.getAttribute('data-theme');
    return forced ? forced === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  };

  // ------------------------------------------------------------ the map
  const OP = {points: 1, areas: 1, wells: 1, base: 1};      // layer opacity, 0..1
  try { Object.assign(OP, JSON.parse(localStorage.getItem('viewer.opacity') || '{}')); } catch (e) { /* storage off */ }
  for (const k of Object.keys(OP)) OP[k] = Math.max(0, Math.min(1, Number(OP[k]) || 0));
  const map = L.map('map', {zoomControl: false}).setView([38.33, -87.55], 10);
  L.control.zoom({position: 'topright'}).addTo(map);
  L.control.scale({imperial: true, position: 'bottomright'}).addTo(map);
  const bases = {
    usgs: L.tileLayer('https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}', {maxZoom: 19, maxNativeZoom: 16, attribution: 'USDA, USGS The National Map: Orthoimagery'}),
    imagery: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom: 19, attribution: 'Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community'}),
    osm: L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}),
    topo: L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {maxZoom: 17, subdomains: 'abc', attribution: '© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)'}),
    grey: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {maxZoom: 16, attribution: 'Esri, HERE, Garmin, (c) OpenStreetMap contributors, and the GIS user community'}),
  };
  let base = bases.usgs.addTo(map);
  $('base').onchange = e => { map.removeLayer(base); base = bases[e.target.value].addTo(map); base.setOpacity(OP.base); };
  map.on('mousemove', e => { $('readout').textContent = `${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)}`; });
  map.createPane('areas').style.zIndex = 450;          // above the points' canvas (overlay pane, 400)
  map.createPane('wells').style.zIndex = 620;          // above the markers
  const outlineLayer = L.layerGroup().addTo(map);
  const drawn = new L.FeatureGroup().addTo(map);
  map.addControl(new L.Control.Draw({
    position: 'topright', edit: {featureGroup: drawn, edit: false, remove: false},
    draw: {polygon: {allowIntersection: false, showArea: true, shapeOptions: {color: '#d97706', weight: 2.5, fill: false}},
           rectangle: {shapeOptions: {color: '#d97706', weight: 2.5, fill: false}},
           polyline: false, circle: false, marker: false, circlemarker: false},
  }));
  let drawing = false;
  map.on(L.Draw.Event.DRAWSTART, () => { drawing = true; });
  map.on(L.Draw.Event.DRAWSTOP, () => setTimeout(() => { drawing = false; }, 50));

  // ------------------------------------------------------------ colours (the KMZ's)
  // ColorBrewer RdBu (red = negative = sinking / away from the satellite)
  const RDBU = [[103,0,31],[178,24,43],[214,96,77],[244,165,130],[253,219,199],[247,247,247],[209,229,240],[146,197,222],[67,147,195],[33,102,172],[5,48,97]];
  const BLUES = [[222,235,247],[198,219,239],[158,202,225],[107,174,214],[66,146,198],[33,113,181],[8,81,156],[8,48,107]];
  function ramp(stops, x) {
    x = Math.max(0, Math.min(1, x)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
    return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
  }
  const SNR_BINS = [[0, 1, [204,204,204], 'below 1'], [1, 1.96, [254,224,139], '1 to 1.96'],
                    [1.96, 3, [244,109,67], '1.96 to 3: stands out'], [3, 1e9, [165,0,38], '3 or more: stands out clearly']];
  const N_RAMP = 41;
  /* client_kmz.nice_limit: the 98th percentile of |v|, up to 1, 2, 2.5 or 5 x 10^k */
  function niceLimit(v) {
    const a = [];
    for (const x of v) if (isFinite(x)) a.push(Math.abs(x));
    if (!a.length) return 1;
    const x = C.percentile(a, 98);
    if (!(x > 0)) return 1;
    const p = 10 ** Math.floor(Math.log10(x));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= x) return m * p;
    return 10 * p;
  }

  // ------------------------------------------------------------ state
  let P = null;               // the project: {B, meta, a, n, T, dates, t, sx, sy, ...}
  const FOOT0 = (document.getElementById('foot') || {}).textContent || '';   // a single project's footer
  let sel = -1;               // the selected point
  let polys = [];             // the polygons on the map: {id, name, rings [[lon, lat], ...], layer, idx}
  let selPoly = null;         // the polygon shown in the panel (one of polys), or null
  let polySeq = 0;
  let pins = [];              // what the compare chart draws, at most 4: {kind: 'pt', j} | {kind: 'poly', id}
  const MAX_PINS = 4;         // the categorical colours that stay distinguishable (viewer_charts.js)
  let units = 'in';
  let chartMode = 'vert';
  // a combined project (combine_viewer) shows east-west where a single
  // project shows line of sight: these say which
  const COMB = () => !!(P && P.meta && P.meta.combined);
  const LOSW = () => COMB() ? 'East-west (indicative)' : 'Line of sight';
  const LOSw = () => COMB() ? 'east-west (indicative)' : 'line of sight';
  const LOSPLUS = () => COMB() ? '+ = east' : '+ = towards the satellite';
  // east-west: red = east, blue = west, as the app's map (line of sight: red = away)
  const LOSRAMP = () => COMB() ? RDBU.slice().reverse() : RDBU;
  function comboLabels() {                     // the layer buttons' words for this project
    const lab = v => { const i = document.querySelector(`#layers input[value="${v}"]`); return i ? i.parentNode : null; };
    const words = COMB() ? {los: ' East-west movement (indicative)', change: ' East-west change, at the average rate'}
      : {los: ' Line-of-sight velocity', change: ' Change over these dates'};
    for (const [v, w] of Object.entries(words)) { const l = lab(v); if (l) l.lastChild.textContent = w; }
    const u = lab('unwrap'); if (u) u.hidden = COMB();
    if (COMB() && layer() === 'unwrap') document.querySelector('#layers input[value="vert"]').checked = true;
  }
  const VERTNOTE = () => COMB() ? 'Vertical from both passes together (east-west separated out)'
    : 'Vertical (line of sight / its up component; assumes all the movement is vertical)';
  const held = {};            // colour limits held while the dates change

  const unitDiv = () => (units === 'in' ? MM_IN : 1);
  const ndRate = () => (units === 'in' ? 2 : 1);
  const fmtRate = mm => (mm === null || !isFinite(mm) ? '—' : `${C.signed(mm / unitDiv(), ndRate())} ${units}/yr`);
  const fmtLen = mm => (mm === null || !isFinite(mm) ? '—' : `${C.signed(mm / unitDiv(), ndRate())} ${units}`);
  const bothRate = mm => (mm === null || !isFinite(mm) ? '—' : units === 'in'
    ? `${C.signed(mm / MM_IN, 2)} in/yr (${C.signed(mm, 1)} mm)` : `${C.signed(mm, 1)} mm/yr (${C.signed(mm / MM_IN, 2)} in)`);
  const bothLen = mm => (mm === null || !isFinite(mm) ? '—' : units === 'in'
    ? `${C.signed(mm / MM_IN, 2)} in (${C.signed(mm, 0)} mm)` : `${C.signed(mm, 0)} mm (${C.signed(mm / MM_IN, 2)} in)`);

  // ------------------------------------------------------------ the points layer
  const PointsLayer = L.Layer.extend({
    onAdd(m) {
      this._c = L.DomUtil.create('canvas', 'leaflet-layer');
      this._c.style.position = 'absolute'; this._c.style.pointerEvents = 'none';
      m.getPanes().overlayPane.appendChild(this._c);
      m.on('moveend resize zoomend viewreset', this.redraw, this);
      m.on('zoomstart', () => { this._c.style.opacity = 0; }, this);
      this._c.style.opacity = OP.points;
      this.redraw();
    },
    onRemove(m) { this._c.remove(); m.off('moveend resize zoomend viewreset', this.redraw, this); },
    redraw() {
      const m = this._map; if (!m) return;
      const size = m.getSize(), dpr = window.devicePixelRatio || 1, c = this._c;
      L.DomUtil.setPosition(c, m.containerPointToLayerPoint([0, 0]));
      c.width = size.x * dpr; c.height = size.y * dpr; c.style.width = size.x + 'px'; c.style.height = size.y + 'px';
      c.style.opacity = OP.points;
      const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, size.x, size.y);
      if (!P || !P.col) return;
      const z = m.getZoom(), scale = 256 * 2 ** z, org = m.getPixelOrigin(), off = m.containerPointToLayerPoint([0, 0]);
      const ox = org.x + off.x, oy = org.y + off.y;
      if (cellsShown()) { this.drawCells(g, size, scale, ox, oy); return; }
      const mpp = 40075016.686 * Math.cos(m.getCenter().lat * Math.PI / 180) / scale;
      const r = Math.max(1.6, Math.min(6, 0.42 * (P.spacing / mpp)));
      const X = P.sx, Y = P.sy, col = P.col, alpha = P.alpha, n = P.n, pal = P.pal;
      P.screen = new Float32Array(2 * n);
      const groups = new Map();
      for (let i = 0; i < n; i++) {
        const x = X[i] * scale - ox, y = Y[i] * scale - oy;
        P.screen[2 * i] = x; P.screen[2 * i + 1] = y;
        if (x < -8 || y < -8 || x > size.x + 8 || y > size.y + 8 || col[i] < 0) continue;
        const k = col[i] * 2 + alpha[i];
        let a = groups.get(k); if (!a) groups.set(k, a = []); a.push(x, y);
      }
      for (const [k, a] of groups) {
        const ci = k >> 1, faded = k & 1, rgb = pal[ci];
        if (ci === P.fastIndex) {
          g.fillStyle = 'rgba(128,128,128,.9)';
          for (let j = 0; j < a.length; j += 2) g.fillRect(a[j] - r * .7, a[j + 1] - r * .7, r * 1.4, r * 1.4);
          continue;
        }
        g.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${faded ? .38 : .95})`;
        const rr = faded ? r * .8 : r;
        g.beginPath();
        for (let j = 0; j < a.length; j += 2) { g.moveTo(a[j] + rr, a[j + 1]); g.arc(a[j], a[j + 1], rr, 0, 6.2832); }
        g.fill();
      }
      const ring = (j, color) => {
        const x = P.screen[2 * j], y = P.screen[2 * j + 1];
        g.lineWidth = 2.5; g.strokeStyle = '#111'; g.beginPath(); g.arc(x, y, r + 4, 0, 6.2832); g.stroke();
        g.lineWidth = 1.5; g.strokeStyle = color; g.beginPath(); g.arc(x, y, r + 4, 0, 6.2832); g.stroke();
      };
      const th = K.theme(darkMode() ? 'dark' : 'light');
      pins.forEach((q, k) => { if (q.kind === 'pt') ring(q.j, th.series[k]); });
      if (sel >= 0) ring(sel, '#fff');
      this._r = r;
    },
    /* every measured 15 m cell in view, as squares of its own size */
    drawCells(g, size, scale, ox, oy) {
      const Q = CELLS, n = Q.n;
      const hx = Math.max(0.6, Math.abs(Q.dx) / 360 * scale / 2);
      const lat = this._map.getCenter().lat, hy = Math.max(0.6, Math.abs(mercY(lat + Math.abs(Q.dy) / 2) - mercY(lat - Math.abs(Q.dy) / 2)) * scale / 2);
      Q.screen = new Float32Array(2 * n); Q.hx = hx; Q.hy = hy;
      const groups = new Map();
      for (let i = 0; i < n; i++) {
        const x = Q.sx[i] * scale - ox, y = Q.sy[i] * scale - oy;
        Q.screen[2 * i] = x; Q.screen[2 * i + 1] = y;
        if (x < -hx || y < -hy || x > size.x + hx || y > size.y + hy || Q.col[i] < 0) continue;
        const k = Q.col[i] * 2 + Q.alpha[i];
        let a = groups.get(k); if (!a) groups.set(k, a = []); a.push(x, y);
      }
      for (const [k, a] of groups) {
        const ci = k >> 1, faded = k & 1, rgb = Q.pal[ci];
        g.fillStyle = ci === Q.fastIndex ? 'rgba(128,128,128,.9)' : `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${faded ? .38 : .9})`;
        for (let j = 0; j < a.length; j += 2) {               // snapped to whole pixels: no seams between cells
          const x0 = Math.round(a[j] - hx), y0 = Math.round(a[j + 1] - hy);
          g.fillRect(x0, y0, Math.round(a[j] + hx) - x0, Math.round(a[j + 1] + hy) - y0);
        }
      }
      const cs = P.cellSel;
      if (cs && cs.config === P.config) {
        const x = mercX(cs.lon) * scale - ox, y = mercY(cs.lat) * scale - oy;
        g.lineWidth = 3; g.strokeStyle = '#111'; g.strokeRect(x - hx - 2, y - hy - 2, 2 * hx + 4, 2 * hy + 4);
        g.lineWidth = 1.5; g.strokeStyle = '#fff'; g.strokeRect(x - hx - 2, y - hy - 2, 2 * hx + 4, 2 * hy + 4);
      }
    },
    nearestCell(pt) {
      const Q = CELLS;
      if (!Q || !Q.screen) return -1;
      let best = -1, bd = Infinity;
      for (let i = 0; i < Q.n; i++) {
        if (Q.col[i] < 0) continue;
        const dx = Q.screen[2 * i] - pt.x, dy = Q.screen[2 * i + 1] - pt.y;
        if (Math.abs(dx) > Q.hx + 1 || Math.abs(dy) > Q.hy + 1) continue;
        const d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    },
    nearest(pt) {
      if (!P || !P.screen || !P.col) return -1;
      let best = -1, bd = Math.max(7, (this._r || 3) + 4) ** 2;
      for (let i = 0; i < P.n; i++) {
        if (P.col[i] < 0) continue;
        const dx = P.screen[2 * i] - pt.x, dy = P.screen[2 * i + 1] - pt.y, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    },
  });
  const points = new PointsLayer().addTo(map);

  // ------------------------------------------------------------ layer opacity (11 Nov 2026)
  // One slider a layer: the points' canvas, the areas' pane, the wells' pane,
  // the basemap's tiles. Remembered on this computer.
  function applyOpacity(k) {
    if (k === 'points' && points._c) points._c.style.opacity = OP.points;
    if (k === 'areas') map.getPane('areas').style.opacity = OP.areas;
    if (k === 'wells') map.getPane('wells').style.opacity = OP.wells;
    if (k === 'base') base.setOpacity(OP.base);
  }
  for (const k of Object.keys(OP)) {
    const el = $('op-' + k), out = $('op-' + k + '-v');
    if (!el) continue;
    el.value = Math.round(OP[k] * 100); out.textContent = el.value + '%';
    el.oninput = () => {
      OP[k] = el.value / 100; out.textContent = el.value + '%'; applyOpacity(k);
      try { localStorage.setItem('viewer.opacity', JSON.stringify(OP)); } catch (e) { /* storage off */ }
    };
    applyOpacity(k);
  }

  const layer = () => document.querySelector('input[name=layer]:checked').value;

  /* Colour index (into pal) and fade flag for n values on layer L_: the
   * points' rule, which the 15 m cells share. s: {vertical, vertical_rel, snr,
   * tc, unwrap, los, change_los, fast}; lim: the line-of-sight scale. */
  function paintValues(L_, n, s, lim) {
    const col = new Int16Array(n), alpha = new Uint8Array(n), fade = $('fade').checked;
    let pal = [];
    // the signal-to-noise is the vertical's: a combined project's east-west
    // (indicative, no tested noise) is never faded by it
    const fadeOff = COMB() && (L_ === 'los' || L_ === 'change');
    const fadeBy = j => (fade && !fadeOff && !(s.snr[j] >= C.Z95) ? 1 : 0);
    if (L_ === 'vert') {
      pal = P.meta.bins.map(b => b[2].map(v => Math.round(v * 255)));
      const v = P.meta.relative_colours ? s.vertical_rel : s.vertical;
      for (let i = 0; i < n; i++) {
        const x = v[i] / MM_IN;
        if (!isFinite(x)) { col[i] = -1; continue; }
        let k = P.meta.bins.findIndex(b => x >= b[0] && x < b[1]); if (k < 0) k = P.meta.bins.length - 1;
        col[i] = k; alpha[i] = fadeBy(i);
      }
    } else if (L_ === 'snr') {
      pal = SNR_BINS.map(b => b[2]);
      for (let i = 0; i < n; i++) {
        const v = s.snr[i];
        if (!isFinite(v)) { col[i] = -1; continue; }
        col[i] = SNR_BINS.findIndex(b => v >= b[0] && v < b[1]); alpha[i] = v < 1 ? 1 : 0;
      }
    } else if (L_ === 'tc' || L_ === 'unwrap') {
      const arr = L_ === 'tc' ? s.tc : s.unwrap, lo = L_ === 'tc' ? 0.4 : 0, hi = 1;
      for (let k = 0; k < N_RAMP; k++) pal.push(ramp(BLUES, k / (N_RAMP - 1)));
      for (let i = 0; i < n; i++) {
        const v = arr[i];
        if (!isFinite(v)) { col[i] = -1; continue; }
        col[i] = Math.round(Math.max(0, Math.min(1, (v - lo) / (hi - lo))) * (N_RAMP - 1));
      }
    } else {
      const arr = L_ === 'los' ? s.los : s.change_los;
      for (let k = 0; k < N_RAMP; k++) pal.push(ramp(LOSRAMP(), k / (N_RAMP - 1)));
      for (let i = 0; i < n; i++) {
        const v = arr[i];
        if (!isFinite(v)) { col[i] = -1; continue; }
        col[i] = Math.round((Math.max(-lim, Math.min(lim, v)) + lim) / (2 * lim) * (N_RAMP - 1));
        alpha[i] = fadeBy(i);
      }
    }
    const fastIndex = pal.length; pal = pal.concat([[128, 128, 128]]);
    for (let i = 0; i < n; i++) if (s.fast[i] && L_ !== 'tc' && L_ !== 'unwrap') { col[i] = fastIndex; alpha[i] = 0; }
    return {col, alpha, pal, fastIndex};
  }

  function colourise() {
    if (!P || !P.W) return;
    const L_ = layer(), n = P.n, a = P.a, W = P.W;
    if (L_ === 'los' || L_ === 'change') {
      const arr = L_ === 'los' ? W.los : W.change_los;
      const lim = $('lockscale').checked && held[L_] ? held[L_] : niceLimit(arr);
      held[L_] = lim;
      P.lim = lim;
    }
    const p = paintValues(L_, n, {vertical: W.vertical, vertical_rel: W.vertical_rel, snr: W.snr, tc: a.tc,
                                  unwrap: a.unwrap, los: W.los, change_los: W.change_los, fast: a.fast}, P.lim);
    // the features' extra points only on the movement layer (the field's grid carries the rest)
    for (let i = 0; i < n; i++) if (a.group[i] > 0 && L_ !== 'vert') p.col[i] = -1;
    Object.assign(P, {col: p.col, alpha: p.alpha, pal: p.pal, fastIndex: p.fastIndex});
    paintCells();
    legend(); points.redraw();
  }

  function legend() {
    const L_ = layer(), g = esc(P.meta.ground || 'farmland'), W = P.W;
    let h = '';
    const span = `${C.fmtDate(W.first)} to ${C.fmtDate(W.last)}`;
    if (L_ === 'vert') {
      h = `<div><b>Vertical movement, in/yr</b>${P.meta.relative_colours ? ` <span class="hint">relative to ordinary ${g}</span>` : ''}</div>`;
      h += P.meta.bins.map(b => `<div><span class="sw" style="background:rgb(${b[2].map(v => Math.round(v * 255)).join(',')})"></span>${esc(b[3])}</div>`).join('');
      $('layerhint').textContent = `${span}. The report map's colours. Faded = within normal variation for a single point over these dates (signal-to-noise below 1.96). Grey square = too fast for this satellite to measure.`;
    } else if (L_ === 'snr') {
      h = `<div><b>Signal-to-noise ratio</b> <span class="hint">|movement| / noise of ordinary ${g}, ${esc(span)}</span></div>`;
      h += SNR_BINS.map(b => `<div><span class="sw" style="background:rgb(${b[2].join(',')})"></span>${esc(b[3])}</div>`).join('');
      $('layerhint').textContent = (W.kind === 'report' && P.meta.snr_chance_pct != null ? `${P.meta.snr_chance_pct.toFixed(1)}% of ordinary ${P.meta.ground || 'farmland'} stands out by chance over the whole record, so only a larger share in an area means anything. ` : '') + 'It says how clearly a point moved, not which way. Signal ÷ threshold = this ÷ 1.96 (1 or more stands out).';
    } else if (L_ === 'tc' || L_ === 'unwrap') {
      const stops = BLUES.map((c, i) => `rgb(${c.join(',')}) ${(i / (BLUES.length - 1) * 100).toFixed(0)}%`).join(',');
      h = `<div><b>${L_ === 'tc' ? 'Measurement quality (temporal coherence)' : 'Unwrapping consistency'}</b></div>
           <div class="bar" style="background:linear-gradient(90deg,${stops})"></div>
           <div class="ticks"><span>${L_ === 'tc' ? '0.4' : '0'}</span><span>1</span></div>`;
      $('layerhint').textContent = L_ === 'tc' ? 'How steadily the radar measured each point over the whole record (1 = perfectly). The same for every date window.'
        : `Below ${(P.meta.gate || 0.7).toFixed(1)}: fails the second quality check - treat that point with extra care. The same for every date window.`;
    } else {
      const lim = P.lim, unit = units === 'in' ? 'in' : 'mm', d = unitDiv(), nd = lim / d < 0.1 ? 2 : 1;
      const stops = LOSRAMP().map((c, i) => `rgb(${c.join(',')}) ${(i / (RDBU.length - 1) * 100).toFixed(0)}%`).join(',');
      h = `<div><b>${L_ === 'los' ? `${LOSW()} velocity, ${unit}/yr` : `${LOSW()} change, ${unit}`}</b> <span class="hint">${esc(span)}</span></div>
           <div class="bar" style="background:linear-gradient(90deg,${stops})"></div>
           <div class="ticks"><span>${C.signed(-lim / d, nd)}</span><span>0</span><span>${C.signed(lim / d, nd)}</span></div>`;
      $('layerhint').textContent = (COMB() ? `East-west from both passes together, relative to ${P.meta.ground || 'the surrounding ground'}; red = moving east, blue = moving west. Indicative: not a tested figure.`
        : `Relative to ordinary ${P.meta.ground || 'farmland'} of the same kind; + = towards the satellite, red = away. Not converted to vertical.`) +
        (L_ === 'change' ? ' First image of these dates to the last: each image carries its own noise, so read the chart too.' : '') +
        ($('lockscale').checked ? ' Colour scale held.' : ' The scale fits these dates (hold it to compare windows).');
    }
    $('legend').innerHTML = h;
  }
  document.querySelectorAll('input[name=layer]').forEach(r => { r.onchange = () => { colourise(); writeHash(); }; });
  $('fade').onchange = colourise;
  $('lockscale').onchange = colourise;
  $('outlines').onchange = e => (e.target.checked ? outlineLayer.addTo(map) : map.removeLayer(outlineLayer));
  $('units').querySelectorAll('button').forEach(b => {
    b.onclick = () => {
      units = b.dataset.v;
      $('units').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
      if (P) { legend(); refreshDetail(); renderCompare(); writeHash(); }
    };
  });

  // ------------------------------------------------------------ the date window
  const KIND_TEXT = {report: 'Report figures (whole record)', look: 'Look, not a tested figure', tested: 'Tested over these dates'};

  function presets() {
    const m = P.meta, t = m.t, d = m.dates, T = d.length, out = [{key: 'all', label: 'Whole record', i0: 0, i1: T - 1}];
    const since = (yrs, label, key) => {
      const i0 = t.findIndex(x => x >= t[T - 1] - yrs);
      if (i0 >= 0 && !C.windowProblem(t, i0, T - 1)) out.push({key, label, i0, i1: T - 1});
    };
    since(1.0, 'Last 12 months', 'last1');
    if (m.recent_window && m.recent_window.length === 2) {
      try {
        const [i0, i1] = C.windowIndex(d, m.recent_window[0], m.recent_window[1]);
        if (!C.windowProblem(t, i0, i1)) out.push({key: 'recent', label: 'Last 2 years (the report\'s recent rate)', i0, i1});
      } catch (e) { /* no such dates */ }
    } else since(2.0, 'Last 2 years', 'last2');
    const y0 = +d[0].slice(0, 4), y1 = +d[T - 1].slice(0, 4);
    for (let y = y1; y >= y0; y--) {
      try {
        const [i0, i1] = C.windowIndex(d, `${y}0101`, `${y}1231`);
        if (!C.windowProblem(t, i0, i1)) out.push({key: `y${y}`, label: String(y), i0, i1});
      } catch (e) { /* no images that year */ }
    }
    return out;
  }

  function setupWindowControls() {
    const m = P.meta, d = m.dates, on = !!m.windows;
    P.presets = presets();
    $('preset').innerHTML = P.presets.map(p => `<option value="${p.key}">${esc(p.label)}</option>`).join('') + '<option value="custom">Custom dates</option>';
    const opts = d.map((x, i) => `<option value="${i}">${C.fmtDate(x)}</option>`).join('');
    $('dstart').innerHTML = opts; $('dend').innerHTML = opts;
    for (const id of ['preset', 'dstart', 'dend']) $(id).disabled = !on;
    $('windownote').hidden = !m.window_note; $('windownote').textContent = m.window_note || '';
  }

  function setWindow(i0, i1, from) {
    const t = P.meta.t;
    const why = C.windowProblem(t, i0, i1);
    if (why) { toast(`These dates cannot be measured: ${why}.`, true); syncWindowControls(); return false; }
    P.i0 = i0; P.i1 = i1;
    document.body.classList.add('busy');
    try {
      P.W = C.windowFigures(P.B, i0, i1, null);
    } catch (e) { toast(e.message, true); document.body.classList.remove('busy'); return false; }
    document.body.classList.remove('busy');
    syncWindowControls(from);
    colourise(); refreshDetail(); renderCompare(); drawTimeline(); writeHash(); requestCells();
    return true;
  }

  function syncWindowControls(from) {
    if (!P || P.i0 === undefined) return;
    const W = P.W, pr = P.presets.find(p => p.i0 === P.i0 && p.i1 === P.i1);
    if (from !== 'preset') $('preset').value = pr ? pr.key : 'custom';
    $('dstart').value = String(P.i0); $('dend').value = String(P.i1);
    const chip = $('kindchip');
    chip.textContent = KIND_TEXT[W.kind]; chip.className = `chip ${W.kind}`;
    chip.title = W.kind === 'look' ? lookNote(W) : W.kind === 'report' ? 'The very figures of the client report, the KMZ and the rasters.' : '';
    $('wsum').textContent = `${W.n_images} images · ${W.years.toFixed(2)} yr`;
  }

  function lookNote(W) {
    const n = Object.values(W.n_sample || {}).reduce((a, b) => a + b, 0);
    return `Over these dates each point's noise is measured on a sample of ${n.toLocaleString()} of the report's ordinary-${P.meta.ground || 'farmland'} cells, not all of them, so its signal-to-noise is a close estimate - a look, not a tested figure. The GNSS regional rate is the whole record's.`;
  }

  $('preset').onchange = e => {
    const p = P.presets.find(x => x.key === e.target.value);
    if (p) setWindow(p.i0, p.i1, 'preset');
  };
  $('dstart').onchange = () => setWindow(+$('dstart').value, Math.max(+$('dstart').value, P.i1));
  $('dend').onchange = () => setWindow(Math.min(P.i0, +$('dend').value), +$('dend').value);

  // the image timeline: a tick per image, the left-out dates in red, gaps of
  // more than 36 days shaded; drag across it to choose the dates
  const TL = {pad: 10};
  function drawTimeline() {
    const svg = $('timeline'), w = svg.clientWidth || 800, h = 34, d = P.meta.dates, T = d.length;
    const day = s => Date.UTC(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8));
    const allDays = d.map(day), drops = Object.keys(P.meta.dropped || {});
    const lo = Math.min(allDays[0], ...drops.map(day)), hi = Math.max(allDays[T - 1], ...drops.map(day));
    const X = v => TL.pad + (v - lo) / Math.max(1, hi - lo) * (w - 2 * TL.pad);
    TL.X = X; TL.days = allDays;
    let s = '';
    for (let i = 1; i < T; i++) {
      if (allDays[i] - allDays[i - 1] > 36 * 864e5) {
        s += `<rect class="gap" x="${X(allDays[i - 1]).toFixed(1)}" y="2" width="${(X(allDays[i]) - X(allDays[i - 1])).toFixed(1)}" height="20"><title>${C.fmtDate(d[i - 1])} to ${C.fmtDate(d[i])}: ${Math.round((allDays[i] - allDays[i - 1]) / 864e5)} days without an image</title></rect>`;
      }
    }
    if (P.i0 !== undefined) {
      s += `<rect class="win" x="${X(allDays[P.i0]).toFixed(1)}" y="1" width="${Math.max(2, X(allDays[P.i1]) - X(allDays[P.i0])).toFixed(1)}" height="22" rx="3"/>`;
    }
    for (let i = 0; i < T; i++) s += `<line class="img" x1="${X(allDays[i]).toFixed(1)}" x2="${X(allDays[i]).toFixed(1)}" y1="6" y2="18"/>`;
    for (const dd of drops) {
      const x = X(day(dd)).toFixed(1);
      s += `<line class="drop" x1="${x}" x2="${x}" y1="3" y2="21" stroke-width="2"><title>${C.fmtDate(dd)}: left out - ${esc(P.meta.dropped[dd])}</title></line>`;
    }
    for (let y = +d[0].slice(0, 4) + 1; y <= +d[T - 1].slice(0, 4); y++) {
      const x = X(Date.UTC(y, 0, 1));
      s += `<text x="${x.toFixed(1)}" y="32" text-anchor="middle">${y}</text>`;
    }
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    svg.innerHTML = s;
  }
  (function brush() {
    const svg = $('timeline');
    let start = null;
    const idxAt = ev => {
      const r = svg.getBoundingClientRect(), x = ev.clientX - r.left;
      let best = 0, bd = Infinity;
      TL.days.forEach((v, i) => { const dd = Math.abs(TL.X(v) - x); if (dd < bd) { bd = dd; best = i; } });
      return best;
    };
    svg.addEventListener('pointerdown', ev => {
      if (!P || !P.meta.windows) return;
      start = idxAt(ev); svg.setPointerCapture(ev.pointerId);
    });
    svg.addEventListener('pointermove', ev => {
      if (start === null) return;
      const k = idxAt(ev), win = svg.querySelector('.win'), a = Math.min(start, k), b = Math.max(start, k);
      if (win) {
        win.setAttribute('x', TL.X(TL.days[a]).toFixed(1));
        win.setAttribute('width', Math.max(2, TL.X(TL.days[b]) - TL.X(TL.days[a])).toFixed(1));
      }
    });
    svg.addEventListener('pointerup', ev => {
      if (start === null) return;
      const k = idxAt(ev), a = Math.min(start, k), b = Math.max(start, k);
      start = null;
      if (b - a >= 1) setWindow(a, b); else drawTimeline();
    });
    window.addEventListener('resize', () => { if (P) drawTimeline(); });
  }());

  // ------------------------------------------------------------ chart cards
  /* A chart in a card: its kind, a hover readout, and PNG / SVG / CSV /
   * table. make(theme) returns {svg, model}; data is {columns, rows, notes}
   * (the CSV and the table view). */
  function chartCard(el, title, make, data) {
    const W = P.W;
    el.innerHTML = `<div class="card"><div class="head"><span class="t"></span><span class="chip ${W.kind}">${esc(KIND_TEXT[W.kind])}</span>
      <button class="small" data-a="png" title="Download as PNG">PNG</button><button class="small" data-a="svg" title="Download as SVG">SVG</button>
      <button class="small" data-a="csv" title="Download the numbers behind it">CSV</button><button class="small" data-a="table" title="Show the numbers">Table</button></div>
      <div class="body"></div><div class="tablewrap" hidden></div></div>`;
    el.querySelector('.t').textContent = title;
    const body = el.querySelector('.body');
    const r = make(K.theme(darkMode() ? 'dark' : 'light'));
    body.innerHTML = r.svg;
    if (r.model) attachHover(body, r.model);
    const fname = ext => C.fileName(P.meta.title, title, W.first, W.last, ext);
    el.querySelector('[data-a=svg]').onclick = () => {
      download(fname('svg'), new Blob([make(K.theme('light')).svg], {type: 'image/svg+xml'}));
    };
    el.querySelector('[data-a=png]').onclick = async () => {
      try { download(fname('png'), await svgToPng(make(K.theme('light')).svg, 2)); }
      catch (e) { toast(`The PNG could not be made (${e.message}); the SVG opens in any browser.`, true); }
    };
    el.querySelector('[data-a=csv]').onclick = () => download(fname('csv'), new Blob([csvText(title, data)], {type: 'text/csv'}));
    const tw = el.querySelector('.tablewrap');
    el.querySelector('[data-a=table]').onclick = () => {
      tw.hidden = !tw.hidden;
      if (!tw.hidden) tw.innerHTML = tableHtml(data);
    };
  }

  /* the lines every download carries: the figure's kind and dates, the
   * report's stamp lines (client_report.stamp_lines) and its numbers */
  function stampLines() {
    const W = P.W, m = P.meta;
    const lines = [`${KIND_TEXT[W.kind]}: ${C.fmtDate(W.first)} to ${C.fmtDate(W.last)}, ${W.n_images} images` +
      (W.kind === 'look' ? '; single-point noise from a sample of the report\'s farmland' : '')];
    return lines.concat(m.stamp || [], [`numbers ${m.numbers_hash || '—'}${m.final ? '' : ' · DRAFT'}` +
      (m.software ? ` · drawn by ${m.software}` : '')]);
  }
  const chartBase = () => ({stamp: stampLines().slice(1), kindText: stampLines()[0], final: !!P.meta.final});

  function csvText(title, data) {
    const head = [`# ${C.formulaSafeText(P.meta.title || '')} - ${C.formulaSafeText(title)}`]
      .concat(stampLines().map(l => `# ${C.formulaSafeText(l)}`), (data.notes || []).map(l => `# ${C.formulaSafeText(l)}`));
    return head.join('\n') + '\n' + [C.csvRow(data.columns)].concat(data.rows.map(r => C.csvRow(r.map(v =>
      (typeof v === 'number' ? (isFinite(v) ? +v.toFixed(5) : '') : v ?? ''))))).join('\n') + '\n';
  }
  function tableHtml(data) {
    const cell = v => (typeof v === 'number' ? (isFinite(v) ? v.toFixed(4) : '') : esc(v ?? ''));
    return `<table class="data"><thead><tr>${data.columns.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>` +
      data.rows.map(r => `<tr>${r.map(v => `<td>${cell(v)}</td>`).join('')}</tr>`).join('') + '</tbody></table>';
  }

  function svgToPng(svg, scale) {
    return new Promise((resolve, reject) => {
      const img = new Image(), url = URL.createObjectURL(new Blob([svg], {type: 'image/svg+xml'}));
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = img.width * scale; c.height = img.height * scale;
        const g = c.getContext('2d'); g.scale(scale, scale); g.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        c.toBlob(b => (b ? resolve(b) : reject(new Error('the browser made no image'))), 'image/png');
      };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('the chart did not load as an image')); };
      img.src = url;
    });
  }

  /* hover: a crosshair that snaps to the nearest image, every series' value
   * at it (series charts); the bar under the pointer (bars) */
  function attachHover(body, model) {
    const svg = body.querySelector('svg'), tip = document.createElement('div');
    tip.className = 'tip'; tip.hidden = true; body.appendChild(tip);
    const NS = 'http://www.w3.org/2000/svg';
    const line = document.createElementNS(NS, 'line');
    line.setAttribute('stroke', '#888'); line.setAttribute('stroke-dasharray', '3 3'); line.setAttribute('visibility', 'hidden');
    svg.appendChild(line);
    const toSvg = ev => { const r = svg.getBoundingClientRect(); return [(ev.clientX - r.left) * model.W / r.width, (ev.clientY - r.top) * model.H / r.height, r]; };
    svg.addEventListener('pointermove', ev => {
      const [x, y, r] = toSvg(ev);
      if (x < model.L || x > model.L + model.w || y < model.T || y > model.T + model.h) { tip.hidden = true; line.setAttribute('visibility', 'hidden'); return; }
      let html = '';
      if (model.kind === 'series') {
        let best = 0, bd = Infinity;
        model.xs.forEach((v, i) => { const dd = Math.abs(v - x); if (dd < bd) { bd = dd; best = i; } });
        line.setAttribute('x1', model.xs[best]); line.setAttribute('x2', model.xs[best]);
        line.setAttribute('y1', model.T); line.setAttribute('y2', model.T + model.h); line.setAttribute('visibility', 'visible');
        html = `<div>${C.fmtDate(model.dates[best])}</div>` + model.series.map((s, k) => {
          const v = s.y[best], th = K.theme(darkMode() ? 'dark' : 'light');
          return `<div><span class="k" style="background:${s.color || th.series[k % th.series.length]}"></span><b>${v === null || !isFinite(v) ? '—' : C.signed(v, model.nd)}</b> ${esc(model.unit)} <span class="hint">${esc(s.name)}</span></div>`;
        }).join('');
      } else if (model.kind === 'bars') {
        const i = Math.floor((x - model.L) / model.w * model.n);
        if (i < 0 || i >= model.n) return;
        const bw = (model.hi - model.lo) / model.n, a0 = model.lo + i * bw;
        html = `<div>${C.signed(a0, 2)} to ${C.signed(a0 + bw, 2)}</div><div><b>${model.share[i].toFixed(1)}%</b> of the points (${model.counts[i]})</div>` +
          (model.cmp ? `<div><b>${model.cmp[i].toFixed(1)}%</b> <span class="hint">of ordinary farmland</span></div>` : '');
      } else if (model.kind === 'cats') {
        const i = Math.floor((x - model.L) / model.w * model.n);
        if (i < 0 || i >= model.n) return;
        html = `<div>${esc(model.cats[i])}</div><div><b>${C.signed(model.values[i], 2)}</b></div>`;
      }
      tip.innerHTML = html; tip.hidden = false;
      const px = ev.clientX - r.left, py = ev.clientY - r.top;
      tip.style.left = `${Math.min(px + 12, r.width - tip.offsetWidth - 4)}px`; tip.style.top = `${Math.max(0, py - tip.offsetHeight - 8)}px`;
    });
    svg.addEventListener('pointerleave', () => { tip.hidden = true; line.setAttribute('visibility', 'hidden'); });
  }

  // ------------------------------------------------------------ the 15 m cells (26 Oct 2026)
  // Zoomed in far enough on the whole record, every measured 15 m cell takes
  // the place of the ~100 m points (results_viewer.api_cells: each cell's own
  // whole-record figures, the client rasters' and the KMZ's); a click gives
  // that cell's balloon and its own chart (api_cell). Date windows stay on
  // the ~100 m points: a window's figures for every cell need the whole cube.
  const CELL_ZOOM = 15;
  // the product grid's posting: 15 m on the Sentinel-1 routes and NISAR, 30 m on DISP-S1
  const cellM = () => Math.round(((P && P.meta && P.meta.resolution) || {}).posting_m || 15);
  const POLYGON_LINES = 300;  // individual series drawn behind a polygon's average (results_viewer.POLYGON_LINES)
  let CELLS = null, cellReq = 0, cellTimer = null;
  const mercX = lon => (lon + 180) / 360;
  const mercY = lat => { const s = Math.sin(lat * Math.PI / 180); return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI); };
  const cellsWanted = () => !STATIC && !!P && !!P.W && P.W.kind === 'report' && !!$('cells15') && $('cells15').checked && map.getZoom() >= CELL_ZOOM;
  const cellsShown = () => cellsWanted() && !!CELLS && CELLS.config === P.config && CELLS.n > 0;

  function updateCellHint() {
    let h = '';
    if (!STATIC && P && $('cells15') && $('cells15').checked) {
      if (P.W && P.W.kind !== 'report') h = `${cellM()} m cells show the whole record only: choose Whole record to see them (or export a site with "KMZ, every ${cellM()} m cell", which follows the dates).`;
      else if (map.getZoom() < CELL_ZOOM) h = `Zoom in to see every ${cellM()} m cell.`;
      else if (CELLS && CELLS.tooMany) h = `${CELLS.tooMany.toLocaleString()} cells in view: zoom in further to see them.`;
      else if (cellsShown()) h = `Every measured ${cellM()} m cell (${CELLS.n.toLocaleString()} around this view); click one for its own figures and chart.`;
    }
    $('cellhint').textContent = h;
  }

  function paintCells() {
    if (!CELLS || !CELLS.n || !P) return;
    Object.assign(CELLS, paintValues(layer(), CELLS.n, CELLS.s, P.lim));
  }

  function requestCells() {
    clearTimeout(cellTimer);
    if (!cellsWanted()) { updateCellHint(); points.redraw(); return; }
    const v = map.getBounds();
    const inside = b => v.getWest() >= b[0] && v.getSouth() >= b[1] && v.getEast() <= b[2] && v.getNorth() <= b[3];
    if (CELLS && CELLS.config === P.config && !CELLS.tooMany && inside(CELLS.bbox)) { updateCellHint(); points.redraw(); return; }
    cellTimer = setTimeout(async () => {
      const id = ++cellReq, config = P.config;
      const px = (v.getEast() - v.getWest()) * 0.25, py = (v.getNorth() - v.getSouth()) * 0.25;
      const bbox = [v.getWest() - px, v.getSouth() - py, v.getEast() + px, v.getNorth() + py];
      $('cellhint').textContent = `Loading the ${cellM()} m cells...`;
      try {
        const r = await api('/api/cells', {config, bbox});
        if (id !== cellReq || !P || P.config !== config) return;
        if (r.too_many) CELLS = {config, bbox, n: 0, tooMany: r.too_many};
        else {
          const f32 = k => new Float32Array(b64bytes(r[k]).buffer), n = r.n;
          const row = new Int32Array(b64bytes(r.row).buffer), col = new Int32Array(b64bytes(r.col).buffer);
          const lat = new Float64Array(n), lon = new Float64Array(n), sx = new Float64Array(n), sy = new Float64Array(n);
          for (let i = 0; i < n; i++) {
            lat[i] = r.y0 + (row[i] + 0.5) * r.dy; lon[i] = r.x0 + (col[i] + 0.5) * r.dx;
            sx[i] = mercX(lon[i]); sy[i] = mercY(lat[i]);
          }
          // gridRow / gridCol: the product grid's (CELLS.col is the colour index, as P.col)
          CELLS = {config, bbox, n, gridRow: row, gridCol: col, lat, lon, sx, sy, dx: r.dx, dy: r.dy,
                   s: {vertical: f32('vertical'), vertical_rel: f32('vertical_rel'), los: f32('los'), change_los: f32('change_los'),
                       snr: f32('snr'), tc: f32('tc'), unwrap: f32('unwrap'), fast: b64bytes(r.fast)}};
          paintCells();
          if (P.pendingCell) {                                // a link's cell (#cell=lat,lon)
            const [la, lo] = P.pendingCell, Q = CELLS;
            P.pendingCell = null;
            for (let i = 0; i < Q.n; i++) {
              if (Math.abs(Q.lat[i] - la) <= Math.abs(Q.dy) / 2 && Math.abs(Q.lon[i] - lo) <= Math.abs(Q.dx) / 2) { selectCell(i); break; }
            }
          }
        }
      } catch (err) { toast(err.message, true); }
      updateCellHint(); points.redraw();
    }, 200);
  }
  map.on('moveend', requestCells);
  if ($('cells15')) $('cells15').onchange = () => { requestCells(); points.redraw(); };

  async function selectCell(k) {
    const Q = CELLS;
    sel = -1; selPoly = null; selWell = null; styleShapes();
    P.cellSel = {config: P.config, row: Q.gridRow[k], col: Q.gridCol[k], lat: Q.lat[k], lon: Q.lon[k]};
    P.cellInfo = null; points.redraw(); showCell(); writeHash();
    try {
      const info = await api('/api/cell', {config: P.config, row: P.cellSel.row, col: P.cellSel.col});
      if (P.cellSel && P.cellSel.row === info.row && P.cellSel.col === info.col) { P.cellInfo = info; showCell(); }
    } catch (err) { toast(err.message, true); }
  }

  function showCell() {
    const c = P.cellSel, info = P.cellInfo, body = $('detailbody');
    $('detailhead').innerHTML = `${cellM()} m cell <span class="chip report">${esc(KIND_TEXT.report)}</span>`;
    let h = `<div class="row"><span class="hint">${c.lat.toFixed(5)}, ${c.lon.toFixed(5)} · this map cell's own figures</span></div>`;
    if (!info) { body.innerHTML = h + '<span class="spin"></span>'; return; }
    if (info.fast) {
      body.innerHTML = h + '<p class="note warn"><b>Not measurable - too fast for this satellite.</b> The ground here changes faster across a few tens of metres than the radar can follow between passes.</p>';
      return;
    }
    h += `<div class="balloon">${info.html}</div><div class="tabs"><button data-m="vert">Vertical</button><button data-m="los">${LOSW()}</button></div><div id="pchart"></div>`;
    body.innerHTML = h;
    body.querySelectorAll('.tabs button').forEach(b => { b.onclick = () => { chartMode = b.dataset.m; drawCellChart(); }; });
    drawCellChart();
  }

  /* the cell's own series (zeroed on its fit at the first image, as a point's)
   * and the KMZ's line (its 3 x 3 cells), over the whole record */
  function drawCellChart(info = P.cellInfo, c = P.cellSel, who = 'this cell') {
    const W = P.W, v = chartMode === 'vert', d = unitDiv(), t = P.meta.t, tw = t.map(x => x - t[0]);
    $('detailbody').querySelectorAll('.tabs button').forEach(b => b.classList.toggle('on', b.dataset.m === chartMode));
    const y = (v ? info.own_vert_mm : info.own_los_mm).map(x => (x === null ? null : x / d));
    const src = (v ? info.vert : info.los).map(x => (x === null ? NaN : x * MM_IN));
    let n = 0, st = 0, sy = 0, stt = 0, sty = 0;
    src.forEach((x, i) => { if (isFinite(x)) { n++; st += tw[i]; sy += x; stt += tw[i] * tw[i]; sty += tw[i] * x; } });
    const b = (n * sty - st * sy) / (n * stt - st * st), a0 = (sy - b * st) / n;
    const anchored = v && P.meta.anchored, bg = anchored ? (P.meta.bg_v || 0) : 0, bgU = anchored ? (P.meta.bg_u || 0) : 0;
    const sd = info.sd_vert_mm_yr;
    const half = isFinite(sd) ? C.Z95 * (v ? Math.hypot(sd, bgU) : sd * info.up) : null;
    const rate = v ? info.rate_vert_mm_yr : info.rate_los_mm_yr;
    const series = [{name: `${who} (its own ${cellM()} m map cell)`, y},
                    {name: '3 x 3 cells around it (the KMZ chart)', y: src.map(x => (isFinite(x) ? (x - a0) / d : null)), width: 1.2, dash: '1 0'}];
    const dates = P.meta.dates;
    const data = {columns: ['date', `${v ? 'vertical' : 'line_of_sight'}_${units}`, `fitted_${units}`, `farmland_95pct_low_${units}`, `farmland_95pct_high_${units}`],
      rows: dates.map((dd, i) => [C.isoDate(dd), y[i] === null ? '' : y[i], rate * tw[i] / d,
        half === null ? '' : (bg * tw[i] - half * tw[i]) / d, half === null ? '' : (bg * tw[i] + half * tw[i]) / d]),
      notes: [`${v ? VERTNOTE() : LOSW()}, relative to ordinary ${P.meta.ground || 'farmland'}${anchored ? ' + the regional GNSS motion' : ''}; zeroed on the straight-line fit at the first image; negative = sinking / away from the satellite`,
              `${cellM()} m cell ${c.lat.toFixed(6)}, ${c.lon.toFixed(6)}`]};
    chartCard($('pchart'), `${v ? 'Vertical' : LOSW()} movement at ${c.lat.toFixed(5)}, ${c.lon.toFixed(5)}`, th => K.seriesChart(Object.assign({
      title: `${v ? 'Vertical' : LOSW()} movement: ${fmtRate(rate)}`,
      subtitle: `${C.fmtDate(W.first)} to ${C.fmtDate(W.last)} · ${cellM()} m cell ${c.lat.toFixed(5)}, ${c.lon.toFixed(5)}`,
      unit: units, dates, series,
      band: half === null ? null : {lo: tw.map(x => (bg * x - half * x) / d), hi: tw.map(x => (bg * x + half * x) / d), label: `ordinary ${P.meta.ground || 'farmland'} at this cell's noise (95%)`},
      fits: isFinite(rate) ? [{name: 'average rate', y0: 0, y1: rate * W.years / d}] : [],
      width: 560, height: 240,
    }, chartBase()), th), data);
  }

  // ------------------------------------------------------------ a point
  map.on('click', e => {
    if (drawing || !P) return;
    const pt = map.latLngToContainerPoint(e.latlng);
    if (cellsShown()) { const k = points.nearestCell(pt); if (k >= 0) selectCell(k); return; }
    const j = points.nearest(pt);
    if (j >= 0) selectPoint(j);
  });

  async function selectPoint(j) {
    sel = j; selPoly = null; P.cellSel = null; selWell = null; styleShapes(); points.redraw();
    P.pointInfo = null;
    showPoint(); writeHash();
    if (!STATIC) {
      try {
        P.pointInfo = await api('/api/point', {config: P.config, j});
        if (sel === j) showPoint();
      } catch (err) { toast(err.message, true); }
    }
  }

  function refreshDetail() {
    if (sel >= 0) showPoint(); else if (selPoly) showPolygon(); else if (P && P.cellSel) showCell();
  }

  function pointRows(j) {
    const W = P.W, a = P.a, g = P.meta.ground || 'farmland', anchored = !!P.meta.anchored;
    const rows = [];
    const v = W.vertical[j], rel = W.vertical_rel[j], up = a.up[j];
    const sdRel = isFinite(W.null[j]) ? W.null[j] / up : NaN, u = W.sd_vertical[j];
    rows.push(['Vertical, per year', `<b>${bothRate(v)}</b>` + (anchored ? '' : ` relative to the surrounding ${esc(g)}`)]);
    if (anchored && isFinite(u)) rows.push(['95% range, this point (with the regional motion)', `${fmtRate(v - C.Z95 * u)} to ${fmtRate(v + C.Z95 * u)}`]);
    if (isFinite(sdRel)) rows.push([`Compared with ordinary ${esc(g)}`, `${fmtRate(rel)}, 95% ${fmtRate(rel - C.Z95 * sdRel)} to ${fmtRate(rel + C.Z95 * sdRel)}`]);
    const stt = W.signal_to_threshold[j];
    if (isFinite(stt)) {
      rows.push(['Signal ÷ threshold', `${C.besideLimit(stt, 1)} ${stt >= 1 ? '(stands out)' : '(within normal variation for a single point)'}`]);
      rows.push(['Signal-to-noise ratio', `${C.besideLimit(stt * C.Z95, C.Z95)} (1.96 or more stands out)`]);
    }
    rows.push([`At the average rate over these dates (${W.years.toFixed(1)} yr)`, bothLen(v * W.years)]);
    if (W.change_los_ends) rows.push(['Vertical change between the ends (30-day means)', bothLen(W.change_los_ends[j] / up + (anchored ? (P.meta.bg_v || 0) * W.years : 0))]);
    rows.push([`${LOSW()}, per year (vs ordinary ${esc(g)})`, `${bothRate(W.los[j])}; ${LOSPLUS()}`]);
    if (W.change_los) rows.push([`${LOSW()}, ${C.fmtDate(W.first)} to ${C.fmtDate(W.last)}`, `${bothLen(W.change_los[j])}, ${COMB() ? 'at the average rate' : 'first image to last'}`]);
    rows.push(['Measurement quality', `${a.tc[j].toFixed(2)} temporal coherence`]);
    const uw = a.unwrap[j], gate = P.meta.gate || 0.7;
    if (isFinite(uw)) rows.push(['Unwrapping consistency', `${C.besideLimit(uw, gate)} ${uw < gate ? `(below ${gate.toFixed(1)}: fails the second quality check - treat this point with extra care)` : '(passes the second quality check)'}`]);
    return rows;
  }

  function showPoint() {
    const j = sel, W = P.W, a = P.a, body = $('detailbody');
    $('detailhead').innerHTML = `Point <span class="chip ${W.kind}">${esc(KIND_TEXT[W.kind])}</span>`;
    if (a.fast[j]) {
      body.innerHTML = `<p class="note warn"><b>Not measurable - too fast for this satellite.</b> The ground here changes faster across a few tens of metres than the radar can follow between passes.</p>`;
      return;
    }
    const balloon = P.pointInfo && P.pointInfo.html;
    let h = `<div class="row"><span class="hint">${a.lat[j].toFixed(5)}, ${a.lon[j].toFixed(5)}</span>
      <button class="small" id="pin">${pinIndex({kind: 'pt', j}) >= 0 ? 'Unpin' : 'Pin to compare'}</button></div>`;
    if (W.kind === 'report' && balloon) {
      h += `<div class="balloon">${balloon}</div>`;
    } else {
      h += `<div class="balloon"><table>${pointRows(j).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table>
        <small>${W.kind === 'look' ? esc(lookNote(W)) : ''} ${esc(footprint())}</small></div>`;
      if (balloon) h += `<details><summary>The report's balloon for this point (whole record)</summary><div class="balloon">${balloon}</div></details>`;
    }
    h += `<div class="tabs"><button data-m="vert">Vertical</button><button data-m="los">${LOSW()}</button></div><div id="pchart"></div>`;
    body.innerHTML = h;
    $('pin').onclick = () => togglePin({kind: 'pt', j});
    body.querySelectorAll('.tabs button').forEach(b => { b.onclick = () => { chartMode = b.dataset.m; drawPointChart(j); }; });
    drawPointChart(j);
  }

  /* posting and resolution, both (client_kmz.point_footer's facts) */
  function footprint() {
    const r = P.meta.resolution || {}, post = r.posting_m, sup = r.support_m;
    if (r.text) return `The point's own map cell. ${r.text}`;          // the report's own words (client_report.json resolution)
    const m = v => `${Math.round(v)} m`;
    const ground = Array.isArray(sup) && sup.length === 2 ? `${m(sup[0])} x ${m(sup[1])}` : isFinite(sup) ? m(sup) : null;
    return `The point's own map cell (${post ? m(post) : `${cellM()} m`} posting)` +
      (ground ? `; each value already combines ground of about ${ground} (the resolution).` : '.');
  }

  function windowDates() { return P.meta.dates.slice(P.i0, P.i1 + 1); }

  function drawPointChart(j) {
    $('detailbody').querySelectorAll('.tabs button').forEach(b => b.classList.toggle('on', b.dataset.m === chartMode));
    const W = P.W, v = chartMode === 'vert', d = unitDiv(), t = P.meta.t, tw = t.slice(P.i0, P.i1 + 1).map(x => x - t[P.i0]);
    let y;
    if (P.a.cell_los) y = C.pointSeries(P.B, j, P.i0, P.i1, v).y.map(x => (x === null ? null : x / d));
    else if (P.pointInfo) y = (v ? P.pointInfo.vert : P.pointInfo.los).map(x => (x === null ? null : x * MM_IN / d));
    else { $('pchart').innerHTML = '<span class="spin"></span>'; return; }
    const anchored = v && P.meta.anchored, bg = anchored ? (P.meta.bg_v || 0) : 0;
    const half = v ? (isFinite(W.sd_vertical[j]) ? C.Z95 * W.sd_vertical[j] : null) : (isFinite(W.null[j]) ? C.Z95 * W.null[j] : null);
    const rate = v ? W.vertical[j] : W.los[j];
    const series = [{name: P.a.cell_los ? 'this point (its own map cell)' : 'the 3 x 3 cells around the point', y}];
    if (P.a.cell_los && P.pointInfo && P.pointInfo.vert && !STATIC) {
      // the KMZ's line: the 3 x 3 cells around the point, on these dates and zeroed the same way
      const src = (v ? P.pointInfo.vert : P.pointInfo.los).slice(P.i0, P.i1 + 1).map(x => (x === null ? NaN : x * MM_IN));
      const ok = src.map(x => isFinite(x));
      let n = 0, st = 0, sy = 0, stt = 0, sty = 0;
      src.forEach((x, i) => { if (ok[i]) { n++; st += tw[i]; sy += x; stt += tw[i] * tw[i]; sty += tw[i] * x; } });
      const b = (n * sty - st * sy) / (n * stt - st * st), a0 = (sy - b * st) / n;
      series.push({name: '3 x 3 cells around it (the KMZ chart)', y: src.map((x, i) => (ok[i] ? (x - a0) / d : null)), width: 1.2, dash: '1 0'});
    }
    const data = {columns: ['date', `${v ? 'vertical' : 'line_of_sight'}_${units}`, `fitted_${units}`, `farmland_95pct_low_${units}`, `farmland_95pct_high_${units}`],
      rows: windowDates().map((dd, i) => [C.isoDate(dd), y[i] === null ? '' : y[i], rate * tw[i] / d,
        half === null ? '' : (bg * tw[i] - half * tw[i]) / d, half === null ? '' : (bg * tw[i] + half * tw[i]) / d]),
      notes: [`${v ? VERTNOTE() : LOSW()}, relative to ordinary ${P.meta.ground || 'farmland'}${anchored ? ' + the regional GNSS motion' : ''}; zeroed on the straight-line fit at the first image; negative = sinking / away from the satellite`,
              `point ${P.a.lat[j].toFixed(6)}, ${P.a.lon[j].toFixed(6)}`]};
    chartCard($('pchart'), `${v ? 'Vertical' : LOSW()} movement at ${P.a.lat[j].toFixed(5)}, ${P.a.lon[j].toFixed(5)}`, th => K.seriesChart(Object.assign({
      title: `${v ? 'Vertical' : LOSW()} movement: ${fmtRate(rate)}`,
      subtitle: `${C.fmtDate(W.first)} to ${C.fmtDate(W.last)} · ${P.a.lat[j].toFixed(5)}, ${P.a.lon[j].toFixed(5)}`,
      unit: units, dates: windowDates(), series,
      band: half === null ? null : {lo: tw.map(x => (bg * x - half * x) / d), hi: tw.map(x => (bg * x + half * x) / d), label: `ordinary ${P.meta.ground || 'farmland'} at this point's noise (95%)`},
      fits: isFinite(rate) ? [{name: 'average rate', y0: 0, y1: rate * W.years / d}] : [],
      width: 560, height: 240,
    }, chartBase()), th), data);
  }

  // ------------------------------------------------------------ compare
  const sameItem = (a, b) => a.kind === b.kind && (a.kind === 'pt' ? a.j === b.j : a.id === b.id);
  const pinIndex = q => pins.findIndex(x => sameItem(x, q));
  const polyById = id => polys.find(p => p.id === id) || null;

  function togglePin(q) {
    const k = pinIndex(q);
    if (k >= 0) pins.splice(k, 1);
    else {
      if (pins.length >= MAX_PINS) { toast(`${MAX_PINS} at most: unpin one first (more colours would not stay distinguishable).`, true); return; }
      pins.push(q);
    }
    points.redraw(); styleShapes(); refreshDetail(); renderCompare(); writeHash();
  }
  $('clearpins').onclick = () => { pins = []; points.redraw(); styleShapes(); renderCompare(); refreshDetail(); writeHash(); };

  /* the rows a polygon averages: its field points, not the too-fast ones */
  const polyRows = p => (p.idx || []).filter(j => !P.a.fast[j]);

  /* a pinned item's label, series (vertical, display units) and rate */
  function pinned(q, k) {
    const W = P.W, d = unitDiv();
    if (q.kind === 'pt') {
      const j = q.j;
      return {label: `${k + 1}: point ${P.a.lat[j].toFixed(4)}, ${P.a.lon[j].toFixed(4)}`, rate: W.vertical[j],
              where: `point ${P.a.lat[j].toFixed(6)}, ${P.a.lon[j].toFixed(6)}`,
              y: C.pointSeries(P.B, j, P.i0, P.i1, true).y.map(x => (x === null ? null : x / d))};
    }
    const p = polyById(q.id), rows = polyRows(p);
    const v = rows.map(j => W.vertical[j]).filter(isFinite);
    return {label: `${k + 1}: ${p.name} (${rows.length} points)`, rate: v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN,
            where: `${p.name}: the plain average of ${rows.length} points; outline ${p.rings[0].map(([x, y]) => `${x.toFixed(5)} ${y.toFixed(5)}`).join(', ')}`,
            y: C.meanSeries(P.B, rows, P.i0, P.i1, true).map(x => (x === null ? null : x / d))};
  }

  function renderCompare() {
    const box = $('compare');
    pins = pins.filter(q => q.kind === 'pt' || polyById(q.id));
    box.hidden = !pins.length || !P;
    if (box.hidden) return;
    const W = P.W, dates = windowDates();
    if (!P.a.cell_los) { $('comparebody').innerHTML = '<p class="empty">Rebuild the client KMZ to compare over any dates.</p>'; return; }
    const th = K.theme(darkMode() ? 'dark' : 'light');
    const items = pins.map((q, k) => pinned(q, k));
    const series = items.map(it => ({name: `${it.label}: ${fmtRate(it.rate)}`, y: it.y}));
    // the list: a colour key, select it, unpin it
    const list = document.createElement('div');
    list.className = 'legend';
    items.forEach((it, k) => {
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = `<span class="sw" style="background:${th.series[k]};border-radius:2px"></span>`;
      const go = document.createElement('a');
      go.href = '#'; go.textContent = `${it.label}: ${fmtRate(it.rate)}`;
      go.onclick = ev => { ev.preventDefault(); const q = pins[k]; if (q.kind === 'pt') selectPoint(q.j); else selectPolygon(polyById(q.id)); };
      const x = document.createElement('button');
      x.className = 'small'; x.textContent = 'Unpin'; x.onclick = () => togglePin(pins[k]);
      row.append(go, x);
      list.appendChild(row);
    });
    const data = {columns: ['date'].concat(items.map((it, k) => `item_${k + 1}_vertical_${units}`)),
      rows: dates.map((dd, i) => [C.isoDate(dd)].concat(series.map(sr => (sr.y[i] === null ? '' : sr.y[i])))),
      notes: items.map((it, k) => `item_${k + 1}: ${it.where}; vertical ${fmtRate(it.rate)}`)
        .concat([`vertical, relative to ordinary ${P.meta.ground || 'farmland'}${P.meta.anchored ? ' + the regional GNSS motion' : ''}; each series zeroed on its own fit at the first image; a polygon is the plain average of its points' own series (a look, not a tested figure)`])};
    const chartEl = document.createElement('div');
    $('comparebody').replaceChildren(chartEl, list);
    chartCard(chartEl, 'Compared, vertical', t => K.seriesChart(Object.assign({
      title: 'Compared: vertical movement', subtitle: `${C.fmtDate(W.first)} to ${C.fmtDate(W.last)}`,
      unit: units, dates, series, width: 560, height: 240}, chartBase()), t), data);
  }

  // ------------------------------------------------------------ polygons
  const SHAPE = {color: '#d97706', weight: 2.5, fill: false};
  const drawOpts = {polygon: {allowIntersection: false, showArea: true, shapeOptions: SHAPE}, rectangle: {shapeOptions: SHAPE}};
  // (a page from before 19 Oct 2026 has no shapes row: the map's own draw buttons remain)
  if ($('drawpoly')) $('drawpoly').onclick = () => new L.Draw.Polygon(map, drawOpts.polygon).enable();
  if ($('drawrect')) $('drawrect').onclick = () => new L.Draw.Rectangle(map, drawOpts.rectangle).enable();

  map.on(L.Draw.Event.CREATED, e => {
    const ll = e.layer.getLatLngs()[0];
    selectPolygon(addPolygon([ll.map(q => [q.lng, q.lat])]));
  });

  /* a polygon on the map, kept beside the others until removed; its outline
   * selects it (the inside stays clickable for the points under it) */
  function addPolygon(rings, name) {
    const p = {id: ++polySeq, name: name || `Polygon ${polySeq}`, rings};
    p.layer = L.polygon(rings.map(r => r.map(([x, y]) => [y, x])), Object.assign({}, SHAPE)).addTo(drawn);
    p.layer.bindTooltip(() => esc(p.name), {sticky: true});
    p.layer.on('click', ev => { L.DomEvent.stopPropagation(ev); if (!drawing) selectPolygon(p); });
    try { p.idx = C.inside(P.a.lon, P.a.lat, P.a.group, rings); } catch (err) { p.idx = []; toast(err.message, true); }
    polys.push(p);
    renderShapes();
    return p;
  }

  function removePolygon(p) {
    p.layer.remove();
    polys = polys.filter(x => x !== p);
    pins = pins.filter(q => !(q.kind === 'poly' && q.id === p.id));
    if (selPoly === p) {
      selPoly = null;
      $('detailhead').textContent = 'Point or polygon';
      $('detailbody').innerHTML = '<p class="empty">Click a point, or draw a polygon.</p>';
    }
    styleShapes(); renderShapes(); renderCompare(); writeHash();
  }

  function selectPolygon(p) {
    if (!p) return;
    sel = -1; selPoly = p; P.cellSel = null; selWell = null;
    styleShapes(); points.redraw(); showPolygon(); writeHash();
  }

  /* outlines: a pinned one in its compare colour, the selected one thicker */
  function styleShapes() {
    const th = K.theme(darkMode() ? 'dark' : 'light');
    for (const p of polys) {
      const k = pinIndex({kind: 'poly', id: p.id});
      p.layer.setStyle({color: k >= 0 ? th.series[k] : SHAPE.color, weight: p === selPoly ? 4 : 2.5,
                        dashArray: p === selPoly || k >= 0 ? null : '6 4'});
    }
    renderShapes();
  }

  /* the shapes row: one chip per polygon (select it), and the draw buttons */
  function renderShapes() {
    const box = $('shapes');
    if (!box) return;
    box.querySelectorAll('.shape').forEach(b => b.remove());
    for (const p of polys) {
      const b = document.createElement('button');
      b.className = 'small shape' + (p === selPoly ? ' primary' : '');
      b.textContent = p.name; b.title = 'Show this polygon';
      b.onclick = () => selectPolygon(p);
      box.appendChild(b);
    }
  }

  /* The polygon panel from every measured 15 m cell inside (26 Oct 2026,
   * results_viewer.api_polygon_cells), fetched once per polygon and dates;
   * the ~100 m points while it comes, and when the polygon is too large. */
  function showPolygon() {
    const W = P.W, poly = selPoly, key = `${W.first}_${W.last}`;
    if (!STATIC) {
      poly.cells = poly.cells || {};
      if (poly.cells[key] === undefined) {
        poly.cells[key] = null;                              // asked for
        api('/api/polygon_cells', {config: P.config, rings: poly.rings, window: {first: W.first, last: W.last}})
          .then(st => { poly.cells[key] = st; }, err => { poly.cells[key] = {error: err.message}; })
          .then(() => { if (selPoly === poly && P.W.first === W.first && P.W.last === W.last) renderPolygon(); });
      }
    }
    renderPolygon();
  }

  function renderPolygon() {
    const W = P.W, g = esc(P.meta.ground || 'farmland');
    const poly = selPoly, idx = poly.idx || [];
    $('detailhead').innerHTML = `<span></span> <span class="chip ${W.kind}">${esc(KIND_TEXT[W.kind])}</span>`;
    $('detailhead').firstChild.textContent = poly.name;
    const cs = !STATIC && poly.cells ? poly.cells[`${W.first}_${W.last}`] : undefined;
    if (cs === null) {
      $('detailbody').innerHTML = `<p class="empty"><span class="spin"></span> Measuring every ${cellM()} m cell inside this polygon...</p>`;
      return;
    }
    const cells = !!(cs && !cs.error && cs.n);
    if (!cells && !idx.length) {
      $('detailbody').innerHTML = `<p class="empty">No measurement ${cs && !cs.error ? 'cells' : 'points'} inside this polygon.</p><div class="row"><button id="premove">Remove this polygon</button></div>`;
      $('premove').onclick = () => removePolygon(poly);
      return;
    }
    const st = cells ? cs : C.polygonStats(P.B, W, idx);
    const what = cells ? 'cells' : 'points';
    const cellNote = cs && cs.error ? `<p class="note warn">${cellM()} m cells: ${esc(cs.error)} These figures are the ~100 m points.</p>` : '';
    const d = unitDiv(), u = units;
    const pct = st.percent_standing_out == null ? '—' : `${st.percent_standing_out.toFixed(0)}%` +
      (W.kind === 'report' && st.percent_by_chance != null ? ` <span class="hint">(ordinary ${g}: ${st.percent_by_chance.toFixed(1)}% by chance)</span>` : '');
    const r = x => (x == null || !isFinite(x) ? '—' : C.signed(x * MM_IN / d, ndRate()));
    $('detailbody').innerHTML = `
      <div class="row"><input type="text" id="aname" aria-label="Name"><button class="small" id="ppin">${pinIndex({kind: 'poly', id: poly.id}) >= 0 ? 'Unpin' : 'Pin to compare'}</button><button class="small" id="premove">Remove</button></div>
      <dl class="stats">
        <dt>${cells ? 'Cells' : 'Points'}</dt><dd>${st.n.toLocaleString()} <span class="hint">${cells ? `every measured ${Math.round(st.cell_m)} m cell inside` : `about ${Math.round(P.spacing)} m apart`}${st.n_too_fast ? `; ${st.n_too_fast} too fast to measure` : ''}</span></dd>
        <dt>Vertical, per year</dt><dd><b>${r(st.vertical_in_yr)} ${u}</b> <span class="hint">average${st.anchored ? '' : `, relative to ordinary ${g}`}</span></dd>
        <dt>Middle 80% of ${what}</dt><dd>${r(st.vertical_in_yr_p10)} to ${r(st.vertical_in_yr_p90)} ${u}/yr</dd>
        <dt>${LOSW()}, per year</dt><dd>${r(st.los_in_yr)} ${u} <span class="hint">vs ordinary ${g}${COMB() ? '; + = east' : ''}</span></dd>
        <dt>${LOSW()}, these dates</dt><dd>${r(st.los_change_in)} ${u} <span class="hint">${COMB() ? 'at the average rate' : 'first image to last'}</span></dd>
        <dt>Standing out</dt><dd>${pct}</dd>
        <dt>Median signal-to-noise</dt><dd>${st.median_snr == null ? '—' : st.median_snr.toFixed(2)}</dd>
      </dl>
      <div class="tabs"><button data-m="vert">Vertical</button><button data-m="los">${LOSW()}</button></div>
      <div id="pchart"></div><div id="hchart"></div><div id="ychart"></div>${cellNote}
      <p class="note">The plain average of the ${what} inside: a look, not a tested figure (neighbouring ${what} share ground).
        For the report's figures - the rate with its 95% range against ordinary ground of the same size, and the
        detection limit - save it as a report area and rebuild the client report.</p>
      <div class="row"><button id="csv">${cells ? `CSV, every ${cellM()} m cell` : 'Points CSV'}</button><button id="geojson">${cells ? 'GeoJSON (~100 m points)' : 'Points GeoJSON'}</button>${STATIC ? '' : `<button id="kmz">KMZ</button><button id="kmzfull" title="Every measured ${cellM()} m cell inside the polygon, over the dates above (up to ~9 km2)">KMZ, every ${cellM()} m cell</button>`}</div>
      ${STATIC ? '' : '<div class="row"><button class="primary" id="save">Save as report area</button><span class="hint">under the name above, as the report should show it</span></div>'}`;
    $('aname').value = poly.name;
    $('aname').oninput = () => {
      poly.name = $('aname').value.trim() || `Polygon ${poly.id}`;
      $('detailhead').firstChild.textContent = poly.name;
      renderShapes(); if (pinIndex({kind: 'poly', id: poly.id}) >= 0) renderCompare(); writeHash();
    };
    $('ppin').onclick = () => togglePin({kind: 'poly', id: poly.id});
    $('premove').onclick = () => removePolygon(poly);
    const draw = async () => {
      $('detailbody').querySelectorAll('.tabs button').forEach(b => b.classList.toggle('on', b.dataset.m === chartMode));
      const v = chartMode === 'vert', dates = windowDates();
      let y;
      if (cells) y = (v ? st.series_vertical_mm : st.series_los_mm).map(x => (x === null ? null : x / d));
      else if (P.a.cell_los) y = C.meanSeries(P.B, st.rows, P.i0, P.i1, v).map(x => (x === null ? null : x / d));
      else {                                               // a table built before date windows: the server's 3 x 3 average
        if (STATIC) return;
        const r0 = await api('/api/polygon', {config: P.config, rings: poly.rings});
        y = (v ? r0.series_vertical_in : r0.series_los_in).map(x => (x === null ? null : x * MM_IN / d));
      }
      const rate = v ? (st.vertical_in_yr ?? NaN) * MM_IN : (st.los_in_yr ?? NaN) * MM_IN;
      // every cell (or point) behind the average, in light grey: the series the average is made of
      let lines = [], of = 0;
      if (cells) { lines = (v ? st.cells_series_vertical_mm : st.cells_series_los_mm) || []; of = st.cells_series_of || lines.length; }
      else if (P.a.cell_los) {
        const rows = st.rows.length <= POLYGON_LINES ? st.rows : Array.from({length: POLYGON_LINES}, (_, k) => st.rows[Math.round(k * (st.rows.length - 1) / (POLYGON_LINES - 1))]);
        lines = rows.map(j => C.pointSeries(P.B, j, P.i0, P.i1, v).y); of = st.rows.length;
      }
      lines = lines.map(ln => ln.map(x => (x === null ? null : x / d)));
      const background = lines.length > 1 ? {lines, label: `each ${cells ? 'cell' : 'point'} (${lines.length < of ? `${lines.length.toLocaleString()} of ${of.toLocaleString()}` : of.toLocaleString()})`} : null;
      chartCard($('pchart'), `Polygon average, ${v ? 'vertical' : LOSw()}`, th => K.seriesChart(Object.assign({
        title: `Average of ${st.n - st.n_too_fast} ${what}, ${v ? 'vertical' : LOSw()}: ${fmtRate(rate)}`,
        subtitle: `${C.fmtDate(W.first)} to ${C.fmtDate(W.last)} · the plain average of each ${cells ? 'cell' : 'point'}'s own series`,
        unit: units, dates, series: [{name: 'polygon average', y}], background,
        fits: isFinite(rate) ? [{name: 'average rate', y0: 0, y1: rate * W.years / d}] : [], width: 560, height: 230,
      }, chartBase()), th), {columns: ['date', `${v ? 'vertical' : 'line_of_sight'}_${units}`], rows: dates.map((dd, i) => [C.isoDate(dd), y[i] === null ? '' : y[i]]),
        notes: [`the plain average of ${st.n - st.n_too_fast} ${what}' own series, each zeroed on its fit at the first image`]});
    };
    $('detailbody').querySelectorAll('.tabs button').forEach(b => { b.onclick = () => { chartMode = b.dataset.m; draw().catch(err => toast(err.message, true)); }; });
    draw().catch(err => toast(err.message, true));
    if (P.a.cell_los && P.a.far) { drawHistogram(st); drawYearly(st); }
    $('csv').onclick = () => exportServer('/api/export_csv', 'csv', st, cells ? {resolution: 'full'} : undefined);
    if (!STATIC) $('kmz').onclick = () => exportServer('/api/export_kmz', 'kmz', st);
    if (!STATIC) $('kmzfull').onclick = () => exportServer('/api/export_kmz', 'kmz', st, {resolution: 'full'});
    $('geojson').onclick = () => exportGeoJSON(st);
    if (!STATIC) $('save').onclick = saveArea;
  }

  /* the polygon's points against ordinary farmland's single points, over
   * the same dates: a share of each, so the two compare */
  function drawHistogram(st) {
    const W = P.W, a = P.a, d = unitDiv();
    const inV = (st.cells_vertical_mm ? st.cells_vertical_mm.filter(x => x !== null) : st.rows.map(j => W.vertical[j])).map(x => x / d).filter(isFinite);
    const farV = [];
    for (let j = 0; j < P.n; j++) if (a.group[j] === 0 && a.far[j] && !a.fast[j] && isFinite(W.vertical[j])) farV.push(W.vertical[j] / d);
    if (inV.length < 3) { $('hchart').innerHTML = ''; return; }
    const lo0 = Math.min(C.percentile(inV, 1), farV.length ? C.percentile(farV, 1) : Infinity);
    const hi0 = Math.max(C.percentile(inV, 99), farV.length ? C.percentile(farV, 99) : -Infinity);
    const step = K.niceStep(hi0 - lo0 || 1, 6), lo = Math.floor(lo0 / step) * step, hi = Math.ceil(hi0 / step) * step;
    const nb = 30, h1 = C.histogram(inV, lo, hi, nb), h2 = farV.length ? C.histogram(farV, lo, hi, nb) : null;
    const mean = inV.reduce((p, q) => p + q, 0) / inV.length;
    const rows = h1.counts.map((c, i) => [lo + i * h1.width, lo + (i + 1) * h1.width, c, h2 ? h2.counts[i] : '']);
    chartCard($('hchart'), 'Point rates against ordinary farmland', th => K.histogramChart(Object.assign({
      title: 'Vertical rate of each point', subtitle: `inside the polygon (bars) and on ordinary ${P.meta.ground || 'farmland'} (line), ${C.fmtDate(W.first)} to ${C.fmtDate(W.last)}`,
      lo, hi, counts: h1.counts, compare: h2 ? h2.counts : null, label: `inside (${inV.length})`,
      compareLabel: `ordinary ${P.meta.ground || 'farmland'} (${farV.length})`, unitLabel: `${units}/yr`,
      marks: [{x: mean, label: `average ${C.signed(mean, ndRate())}`}], width: 560, height: 200}, chartBase()), th),
      {columns: [`from_${units}_yr`, `to_${units}_yr`, 'points_inside', 'farmland_points'], rows,
       notes: [`${h1.below} inside below the range and ${h1.above} above it are not drawn`]});
  }

  function drawYearly(st) {
    const m = P.meta, d = unitDiv();
    const y = st.series_vertical_mm || C.meanSeries(P.B, st.rows, P.i0, P.i1, true);
    const yr = C.yearlyRates(m.dates, m.t, y, P.i0);
    if (!yr.length) { $('ychart').innerHTML = ''; return; }
    chartCard($('ychart'), 'Polygon average, each calendar year', th => K.barsChart(Object.assign({
      title: 'Vertical rate of the average, each calendar year', subtitle: `years with ${C.MIN_DATES}+ images over 3+ months`,
      cats: yr.map(r => String(r.year)), values: yr.map(r => r.rate / d), unitLabel: `${units}/yr`, nd: ndRate(), width: 560, height: 190,
    }, chartBase()), th), {columns: ['year', 'images', `vertical_${units}_yr`], rows: yr.map(r => [String(r.year), r.n_images, r.rate / d]),
      notes: ['the slope of the polygon average over each calendar year; a single year carries seasonal soil and crop signals']});
  }

  async function exportServer(path, ext, st, extra) {
    if (STATIC) { exportCsvStatic(st); return; }
    const name = (($('aname') || {}).value || '').trim() || 'polygon';
    if (extra && extra.resolution === 'full') toast(`Building the ${cellM()} m KMZ from the project's results (the first one takes up to a minute)...`, false, 6000);
    try {
      const r = await api(path, Object.assign({config: P.config, rings: selPoly.rings, name, window: {first: P.W.first, last: P.W.last}}, extra || {}));
      if (ext === 'csv') download(r.filename, new Blob([r.text], {type: 'text/csv'}));
      else download(r.filename, new Blob([b64bytes(r.b64)], {type: 'application/vnd.google-earth.kmz'}));
    } catch (err) { toast(err.message, true); }
  }

  function pointProps(j) {
    const W = P.W, a = P.a, f = x => (isFinite(x) ? +x.toFixed(5) : null);
    return {latitude: +a.lat[j].toFixed(6), longitude: +a.lon[j].toFixed(6),
      vertical_mm_yr: f(W.vertical[j]), line_of_sight_mm_yr: f(W.los[j]),
      line_of_sight_change_mm: W.change_los ? f(W.change_los[j]) : null, signal_to_noise: f(W.snr[j]),
      signal_to_threshold: f(W.signal_to_threshold[j]), temporal_coherence: f(a.tc[j]), too_fast_to_measure: !!a.fast[j]};
  }

  function exportGeoJSON(st) {
    const W = P.W, all = selPoly.idx;
    const gj = {type: 'FeatureCollection',
      properties: {title: P.meta.title, first_image: W.first, last_image: W.last, figures: KIND_TEXT[W.kind], numbers_hash: P.meta.numbers_hash,
                   stamp: stampLines(), units: 'mm and mm/yr; vertical assumes all the movement is vertical; negative = sinking / away from the satellite'},
      features: [{type: 'Feature', properties: {name: selPoly.name}, geometry: {type: 'Polygon', coordinates: selPoly.rings.map(r => r.concat([r[0]]))}}]
        .concat(all.map(j => ({type: 'Feature', properties: pointProps(j), geometry: {type: 'Point', coordinates: [+P.a.lon[j].toFixed(6), +P.a.lat[j].toFixed(6)]}})))};
    download(C.fileName(P.meta.title, `${selPoly.name} points`, W.first, W.last, 'geojson'), new Blob([JSON.stringify(gj)], {type: 'application/geo+json'}));
  }

  function exportCsvStatic(st) {
    const W = P.W, all = selPoly.idx;
    const cols = Object.keys(pointProps(all[0]));
    const data = {columns: cols, rows: all.map(j => cols.map(k => { const v = pointProps(j)[k]; return v === null ? '' : (typeof v === 'boolean' ? (v ? 1 : 0) : v); })),
      notes: ['mm and mm/yr, relative to ordinary farmland of the same kind (+ the regional GNSS motion in the vertical when anchored)']};
    download(C.fileName(P.meta.title, `${selPoly.name} points`, W.first, W.last, 'csv'), new Blob([csvText(`${selPoly.name} points`, data)], {type: 'text/csv'}));
  }

  async function saveArea() {
    const name = ($('aname').value || '').trim();
    if (!name) { toast('Type the name the client report should show for this area.', true); return; }
    if (!confirm(`Add "${name}" to this project's report areas? The config is backed up first.`)) return;
    try { const r = await api('/api/save_area', {config: P.config, rings: selPoly.rings, name}); toast(r.message, false, 9000); }
    catch (err) { toast(err.message, true, 9000); }
  }

  // ------------------------------------------------------------ the view in the address
  /* #project=<name>&from=YYYY-MM-DD&to=YYYY-MM-DD&layer=<layer>&units=in|mm&at=lat,lon,zoom&well=<id>:
   * a link a coworker opens to the same view (the server still asks for its
   * own page token, so the link holds no secret) */
  function readHash() {
    const h = {};
    for (const part of (location.hash || '').replace(/^#/, '').split('&')) {
      const k = part.indexOf('=');
      if (k > 0) h[decodeURIComponent(part.slice(0, k))] = decodeURIComponent(part.slice(k + 1));
    }
    return h;
  }
  function writeHash() {
    if (!P || P.i0 === undefined || STATIC && !history.replaceState) return;
    const c = map.getCenter();
    const parts = [['project', P.name], ['from', C.isoDate(P.W.first)], ['to', C.isoDate(P.W.last)], ['layer', layer()],
                   ['units', units], ['at', `${c.lat.toFixed(5)},${c.lng.toFixed(5)},${map.getZoom()}`],
                   ['pt', sel >= 0 ? `${P.a.lat[sel].toFixed(6)},${P.a.lon[sel].toFixed(6)}` : ''],
                   ['cell', P.cellSel ? `${P.cellSel.lat.toFixed(6)},${P.cellSel.lon.toFixed(6)}` : ''],
                   ['well', selWell ? String(selWell.id) : ''],
                   ['poly', polys.map(p => p.rings[0].map(([x, y]) => `${x.toFixed(5)},${y.toFixed(5)}`).join(';')).join('|')],
                   ['names', polys.some(p => p.name !== `Polygon ${p.id}`) ? polys.map(p => p.name.replace(/\|/g, '/')).join('|') : ''],
                   ['sp', selPoly ? String(polys.indexOf(selPoly)) : ''],
                   ['pins', pins.map(q => (q.kind === 'pt' ? `p${P.a.lat[q.j].toFixed(6)},${P.a.lon[q.j].toFixed(6)}` : `g${polys.findIndex(p => p.id === q.id)}`)).join('|')]]
      .filter(([, v]) => v).map(([k, v]) => `${k}=${encodeURIComponent(v)}`);
    try { history.replaceState(null, '', `#${parts.join('&')}`); } catch (e) { /* a file:// page may refuse */ }
  }
  map.on('moveend', () => writeHash());

  function applyHash(h) {
    if (h.layer) {
      const r = document.querySelector(`input[name=layer][value="${CSS.escape(h.layer)}"]`);
      if (r) r.checked = true;
    }
    if (h.units === 'mm' || h.units === 'in') {
      units = h.units;
      $('units').querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.v === units));
    }
    if (h.at) {
      const [la, lo, z] = h.at.split(',').map(Number);
      if (isFinite(la) && isFinite(lo) && isFinite(z)) map.setView([la, lo], z);
    }
    if ((h.from || h.to) && P.meta.windows) {
      try { const [i0, i1] = C.windowIndex(P.meta.dates, h.from, h.to); setWindow(i0, i1); } catch (e) { toast(e.message, true); }
    } else colourise();
    const nearestTo = txt => {
      const [la, lo] = txt.split(',').map(Number), k = Math.cos(la * Math.PI / 180);
      let best = -1, bd = Infinity;
      for (let j = 0; j < P.n; j++) {
        const d = (P.a.lat[j] - la) ** 2 + ((P.a.lon[j] - lo) * k) ** 2;
        if (d < bd) { bd = d; best = j; }
      }
      return bd < 1e-6 ? best : -1;
    };
    const names = (h.names || '').split('|');
    const made = (h.poly || '').split('|').map((txt, k) => {
      const ring = txt.split(';').map(q => q.split(',').map(Number)).filter(q => q.length === 2 && q.every(isFinite));
      return ring.length >= 3 ? addPolygon([ring], names[k] || undefined) : null;
    });
    for (const tok of (h.pins || '').split('|')) {
      if (pins.length >= MAX_PINS) break;
      if (tok[0] === 'p') { const j = nearestTo(tok.slice(1)); if (j >= 0) pins.push({kind: 'pt', j}); }
      if (tok[0] === 'g' && made[+tok.slice(1)]) pins.push({kind: 'poly', id: made[+tok.slice(1)].id});
    }
    styleShapes(); renderCompare();
    if (h.cell) {                                            // a 15 m cell: selected once the cells in view arrive
      const [la, lo] = h.cell.split(',').map(Number);
      if (isFinite(la) && isFinite(lo)) P.pendingCell = [la, lo];
    } else if (h.pt) { const j = nearestTo(h.pt); if (j >= 0) selectPoint(j); }
    else if (h.sp !== undefined && made[+h.sp]) selectPolygon(made[+h.sp]);
  }

  // ------------------------------------------------------------ wells (11 Nov 2026)
  // The study's point layers (wells) from /api/wells: each well's reading from
  // the client report (whole record, relative to ordinary ground), and, for the
  // dates chosen here, the measurement points around it (a look, not a test).
  const wellLayer = L.layerGroup().addTo(map);
  const wellCanvas = L.canvas({pane: 'wells', padding: 0.3});     // thousands of wells: one canvas, not SVG
  const wellR = () => { const z = map.getZoom(); return z < 12 ? 2 : z < 14 ? 3 : z < 16 ? 4.5 : 6.5; };
  map.on('zoomend', () => wellLayer.eachLayer(m => m.setRadius(wellR())));
  let WELLS = [];                                // [{layer, ...point}] of the project shown
  let selWell = null;                            // the well in the panel (a link names it: #...&well=<id>)
  const WELL_R = 150;                            // metres: "the points around a well"
  $('wellson').onchange = e => (e.target.checked ? wellLayer.addTo(map) : map.removeLayer(wellLayer));
  async function loadWells(config, want) {          // want: a well's id from the link
    wellLayer.clearLayers(); WELLS = []; $('wells').hidden = true;
    if (STATIC || !config) return;
    let d;
    try { d = await api('/api/wells', {config}); } catch (e) { return; }      // an older server: no wells
    if (!P || P.config !== config) return;
    const colour = {};
    for (const L_ of d.layers) for (const c of L_.legend) colour[c.name] = c.color;
    for (const L_ of d.layers) {
      for (const w of L_.points) {
        w.layer = L_.label; w.from_report = L_.from_report; w.colour = colour[w.status_words] || '#888';
        WELLS.push(w);
        L.circleMarker([w.lat, w.lon], {renderer: wellCanvas, radius: wellR(), color: '#111', weight: 1, fillColor: w.colour, fillOpacity: 1})
          .bindTooltip(`${esc(w.id)} · ${esc(w.status_words)}`)
          .on('click', ev => { L.DomEvent.stopPropagation(ev); showWell(w); })
          .addTo(wellLayer);
      }
    }
    if (!WELLS.length) return;
    $('wells').hidden = false;
    $('wellshead').textContent = `${d.layers.map(x => x.label).join(', ')} (${WELLS.length.toLocaleString()})`;
    renderWellSummary(d);
    const w = want && WELLS.find(x => String(x.id) === want);
    if (w) { map.setView([w.lat, w.lon], Math.max(map.getZoom(), 15)); showWell(w); }
  }
  const wellRate = w => (w.value_mm_yr == null ? '—' : fmtRate(w.value_mm_yr));
  function renderWellSummary(d) {
    const counts = new Map();
    for (const w of WELLS) counts.set(w.status_words, (counts.get(w.status_words) || 0) + 1);
    const order = [];
    for (const L_ of d.layers) for (const c of L_.legend) if (!order.some(o => o.name === c.name)) order.push(c);
    const measured = WELLS.filter(w => w.value_mm_yr != null).sort((a, b) => a.value_mm_yr - b.value_mm_yr);
    let h = `<div class="wcounts">${order.map(c => `<i style="background:${esc(c.color)}"></i><span>${esc(c.name)}</span><b>${counts.get(c.name) || 0}</b>`).join('')}</div>`;
    if (measured.length) {
      const med = measured[Math.floor(measured.length / 2)].value_mm_yr;
      h += `<p class="hint">${measured.length} with a number; median ${esc(fmtRate(med))}, fastest sinking ${esc(fmtRate(measured[0].value_mm_yr))}. Whole record, relative to ordinary ground.</p>`;
    } else if (!WELLS.some(w => w.from_report)) {
      h += '<p class="hint">No readings yet: the products run gives each well its number.</p>';
    }
    const rows = WELLS.slice().sort((a, b) => (b.status === 'moving') - (a.status === 'moving') ||
      (a.value_mm_yr ?? Infinity) - (b.value_mm_yr ?? Infinity));
    h += `<div class="wlist"><table>${rows.map((w, k) => `<tr data-k="${WELLS.indexOf(w)}"><td><i class="wdot" style="display:inline-block;width:9px;height:9px;background:${esc(w.colour)};border:1px solid #111"></i> ${esc(w.id)}</td><td class="v">${esc(wellRate(w))}</td></tr>`).join('')}</table></div>`;
    $('wellsum').innerHTML = h;
    $('wellsum').querySelectorAll('tr[data-k]').forEach(tr => {
      tr.onclick = () => { const w = WELLS[+tr.dataset.k]; map.setView([w.lat, w.lon], Math.max(map.getZoom(), 15)); showWell(w); };
    });
  }
  /* the measurement points within WELL_R of a well, over the dates chosen: their
     count, median and spread of vertical rate, and the nearest one */
  function aroundWell(w) {
    if (!P || !P.W) return null;
    const k = Math.cos(w.lat * Math.PI / 180), m = 111320, idx = [];
    let near = -1, nd = Infinity;
    for (let j = 0; j < P.n; j++) {
      const dy = (P.a.lat[j] - w.lat) * m, dx = (P.a.lon[j] - w.lon) * m * k, d = Math.hypot(dx, dy);
      if (d < nd && !P.a.fast[j]) { nd = d; near = j; }
      if (d <= WELL_R && !P.a.fast[j] && isFinite(P.W.vertical[j])) idx.push(j);
    }
    const v = idx.map(j => P.W.vertical[j]).sort((a, b) => a - b);
    const q = f => v[Math.min(v.length - 1, Math.max(0, Math.round(f * (v.length - 1))))];
    const stand = idx.filter(j => P.W.signal_to_threshold[j] >= 1).length;
    return {n: v.length, med: v.length ? q(0.5) : NaN, lo: v.length ? q(0.1) : NaN, hi: v.length ? q(0.9) : NaN,
            stand, near, near_m: nd};
  }
  function showWell(w) {
    sel = -1; selPoly = null; if (P) P.cellSel = null; styleShapes(); points.redraw();
    selWell = w;
    const n = x => (x == null ? '' : C.signed(x / unitDiv(), ndRate()));
    const rows = [];
    rows.push(['Reading (client report)', `<b>${esc(w.status_words)}</b>`]);
    if (w.value_mm_yr != null) rows.push(['Vertical, per year (whole record)', `<b>${esc(bothRate(w.value_mm_yr))}</b> relative to ordinary ground`]);
    if (w.range_mm_yr) rows.push(['95% range', `${n(w.range_mm_yr[0])} to ${n(w.range_mm_yr[1])} ${units}/yr`]);
    if (w.tier) rows.push(['Tier', esc(w.tier)]);
    if ((w.shared_with || []).length) rows.push(['Shared value', `with ${esc(w.shared_with.join(', '))}: not a per-well rate`]);
    if (w.reason) rows.push(['Why', esc(w.reason)]);
    for (const [k, v] of Object.entries(w.attributes || {})) if (v != null && v !== '') rows.push([esc(k.replace(/_/g, ' ')), esc(v)]);
    const A = aroundWell(w);
    if (A) {
      const W = P.W;
      rows.push([`Points within ${WELL_R} m, ${C.fmtDate(W.first)} to ${C.fmtDate(W.last)}`,
        A.n ? `${A.n}; median ${esc(fmtRate(A.med))} (middle 80% ${esc(fmtRate(A.lo))} to ${esc(fmtRate(A.hi))}); ${A.stand} stand out` : 'none measured']);
      if (A.near >= 0) rows.push(['Nearest point', `${Math.round(A.near_m)} m away`]);
    }
    $('detailhead').innerHTML = `Well ${esc(w.id)}`;
    $('detailbody').innerHTML = `<div class="row"><span class="hint">${w.lat.toFixed(5)}, ${w.lon.toFixed(5)} · ${esc(w.layer)}</span></div>
      <div class="balloon"><table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table>
      <small>The reading and range are the client report's, over the whole record. The points around the well are a look over the dates chosen here, not a tested figure.</small></div>
      <div class="tabs"><button data-m="vert">Vertical</button><button data-m="los">${LOSW()}</button></div>
      <p class="hint" id="wcellnote"></p><div id="pchart"><span class="spin"></span></div>`;
    writeHash();
    if (STATIC) { $('pchart').innerHTML = '<p class="empty">The well\'s own series needs the live viewer.</p>'; return; }
    const draw = info => {
      $('detailbody').querySelectorAll('.tabs button').forEach(b => { b.onclick = () => { chartMode = b.dataset.m; draw(info); }; });
      if (info.fast) { $('pchart').innerHTML = '<p class="note warn"><b>Not measurable here - too fast for this satellite.</b></p>'; return; }
      drawCellChart(info, {lat: info.lat, lon: info.lon}, 'the well');
    };
    const cached = w.cell;
    const go = info => {
      if (selWell !== w) return;
      $('wcellnote').textContent = info.offset_m <= cellM() / 2 + 1
        ? `The well's own ${cellM()} m map cell, whole record (the date window does not cut this chart).`
        : `The nearest measured ${cellM()} m map cell, ${Math.round(info.offset_m)} m from the well (its own cell has no measurement); whole record.`;
      draw(info);
    };
    if (cached) { go(cached); return; }
    api('/api/well', {config: P.config, lat: w.lat, lon: w.lon})
      .then(info => { w.cell = info; go(info); },
            err => { if (selWell === w) $('pchart').innerHTML = `<p class="empty">${esc(err.message)}</p>`; });
  }

  // ------------------------------------------------------------ loading
  async function projects() {
    if (STATIC) {
      $('projgroup').hidden = true;
      return load(null);
    }
    const d = await api('/api/projects');
    const s = $('project');
    s.innerHTML = d.projects.map(p => `<option value="${esc(p.config)}" ${p.ready ? '' : 'disabled'}>${esc(p.name)}${p.ready ? ` — points built ${esc(p.built)}` : ' — no viewer points yet'}</option>`).join('');
    let want = null; try { want = localStorage.getItem('viewer.project'); } catch (e) { /* storage off */ }
    const ready = d.projects.filter(p => p.ready);
    const h = readHash();
    const pick = (h.project && ready.find(p => p.name === h.project)) || ready.find(p => p.config === want) || ready[0];
    if (pick) { s.value = pick.config; load(pick.config, pick.name, h.project === pick.name ? h : null); }
    else toast('No project has viewer points yet: build the client KMZ (Products in the dashboard) first.', true, 9000);
  }
  const nameOf = config => { const o = [...$('project').options].find(x => x.value === config); return o ? o.textContent.split(' — ')[0] : ''; };
  $('project').onchange = e => load(e.target.value, nameOf(e.target.value));
  $('reload').onclick = () => load($('project').value, nameOf($('project').value));

  async function load(config, name, hash) {
    $('wsum').innerHTML = '<span class="spin"></span>';
    try {
      const B = STATIC ? await window.INSAR_BUNDLE_STATIC() : C.decodeBundle(await apiBinary('/api/bundle', {config}));
      if (!STATIC) { try { localStorage.setItem('viewer.project', config); } catch (e) { /* storage off */ } }
      const m = B.meta, a = B.arrays, n = m.n;
      const sx = new Float64Array(n), sy = new Float64Array(n);
      for (let i = 0; i < n; i++) {                      // web mercator at zoom 0
        sx[i] = (a.lon[i] + 180) / 360;
        const s = Math.sin(a.lat[i] * Math.PI / 180);
        sy[i] = 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
      }
      P = {config, name: name || '', B, meta: m, a, n, T: m.dates.length, sx, sy, spacing: m.spacing_m || 100, cellSel: null};
      CELLS = null;
      comboLabels();
      { const lab = $('cells15') && $('cells15').parentNode; if (lab) lab.lastChild.textContent = ` Every ${cellM()} m cell when zoomed in`; }
      if (m.windows) {                                    // the load check: the own-cell series give the stored figures back
        const chk = C.wholeRecordCheck(B);
        if (!(chk.rate <= 0.01 && chk.change <= 0.01)) {
          m.windows = false;
          m.window_note = `The points' own series do not reproduce the delivered figures (rate ${chk.rate}, change ${chk.change}): date windows are off. Rebuild the client KMZ.`;
        }
      }
      sel = -1; pins = []; polys = []; selPoly = null; drawn.clearLayers(); renderShapes();
      document.title = `${m.title || 'Results'} - results viewer`;
      $('stale').hidden = !m.stale; $('stale').textContent = m.stale || '';
      const rb = $('routebadge');
      rb.hidden = m.route === 'official';
      rb.textContent = m.combined ? 'Combined view (review)' : m.screening ? 'Screening only' : 'Comparison only';
      rb.title = m.combined ? 'Both passes together: the vertical awaits sign-off as a client figure; east-west is indicative.'
        : 'Not the official route: these figures are for comparison or screening, not for a client.';
      $('foot').textContent = m.combined
        ? `Both passes together (${(m.sources || []).map(x => `${x.project}, ${x.direction}`).join(' and ')}): every whole-record figure is the combined result's, the same numbers as its rasters. Each point's series is solved date by date from both passes (the descending interpolated onto the ascending dates). Negative vertical = sinking; positive east-west = moving east.`
        : FOOT0;
      const fb = $('finalbadge');
      fb.hidden = false; fb.textContent = m.final ? 'FINAL' : 'DRAFT'; fb.className = `badge ${m.final ? 'final' : 'draft'}`;
      fb.title = m.final ? 'Signed off: these are the released numbers.' : `Not signed off${(m.blockers || []).length ? `: ${m.blockers.join('; ')}` : ''}`;
      outlineLayer.clearLayers();
      for (const ar of m.areas || []) L.polygon(ar.rings.map(r => r.map(([x, y]) => [y, x])), {pane: 'areas', color: '#1c2530', weight: 2, fill: false, interactive: false}).bindTooltip(ar.name).addTo(outlineLayer);
      for (const f of m.features || []) L.polygon(f.rings.map(r => r.map(([x, y]) => [y, x])), {pane: 'areas', color: '#8b1a1a', weight: 2, fillOpacity: .08, interactive: false}).addTo(outlineLayer);
      loadWells(config, hash && hash.well);
      const [w, s, e, nn] = m.bbox; map.fitBounds([[s, w], [nn, e]], {padding: [20, 20]});
      setupWindowControls();
      $('detailhead').textContent = 'Point or polygon';
      $('detailbody').innerHTML = '<p class="empty">Click a point for its figures and its movement over these dates, or draw a polygon around a place (the buttons above, or the shapes at the top right of the map) for the average of the points inside it.</p>';
      setWindow(0, P.T - 1, 'load');
      renderCompare();
      if (hash) applyHash(hash);
      writeHash();
      toast(`${n.toLocaleString()} points, one every ~${Math.round(P.spacing)} m; ${P.T} images`);
    } catch (e) { $('wsum').textContent = ''; toast(e.message, true, 9000); }
  }

  projects().catch(e => toast(e.message, true, 9000));
}());
