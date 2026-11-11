/* The pipeline app's page (app_server.py). Everything shown comes from the
 * server's API, which reads what is on disk; this file only draws it.
 * Text from the server is always set as text (h() below), never as HTML. */
(function () {
  'use strict';
  const TOKEN = (document.querySelector('meta[name="insar-token"]') || {}).content || '';
  const $ = (s, el) => (el || document).querySelector(s);
  const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'style') el.style.cssText = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const c of kids.flat()) {
      if (c == null || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  // replaceChildren() would print a null child as the word "null"
  function put(el, ...kids) {
    el.replaceChildren(...kids.flat().filter(k => k != null && k !== false));
  }

  // ------------------------------------------------------------ API
  let online = true;
  async function api(path, body) {
    let r;
    try {
      r = await fetch('/api/' + path, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Insar-Token': TOKEN},
                                       body: JSON.stringify(body || {})});
    } catch (e) {
      setOnline(false);
      throw new Error('The app has stopped on this computer. Open it again from the InSAR Pipeline icon.');
    }
    setOnline(true);
    const j = await r.json().catch(() => ({error: 'bad answer from the server'}));
    if (!r.ok) throw new Error(j.error || ('error ' + r.status));
    return j;
  }
  function setOnline(on) {
    if (on === online) return;
    online = on;
    $('#conn').classList.toggle('off', !on);
    $('#conn span').textContent = on ? 'connected' : 'not connected';
  }
  function toast(msg, tone) {
    const t = h('div', {class: 'toast ' + (tone || '')}, msg);
    $('#toasts').append(t);
    setTimeout(() => t.remove(), tone === 'bad' ? 9000 : 5000);
  }
  function confirmBox(title, text, yes) {
    const d = $('#dconfirm');
    $('#cftitle').textContent = title;
    $('#cftext').textContent = text;
    $('#cfyes').textContent = yes || 'OK';
    d.returnValue = '';
    d.showModal();
    return new Promise(res => d.addEventListener('close', () => res(d.returnValue === 'yes'), {once: true}));
  }

  // ------------------------------------------------------------ theme
  try {
    const t = localStorage.getItem('insar-app-theme');
    if (t) document.documentElement.dataset.theme = t;
  } catch (e) { /* no storage: the OS setting */ }
  $('#theme').addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('insar-app-theme', next); } catch (e) { /* fine */ }
    const f = $('#rframe');                       // the results section's viewer follows
    if (f && f.src && f.contentWindow) { try { f.contentWindow.postMessage({theme: next}, new URL(f.src).origin); } catch (e) { /* not loaded */ } }
  });

  // ------------------------------------------------------------ maps
  const TILES = {
    imagery: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
              {maxZoom: 19, attribution: 'Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community'}],
    osm: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          {maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}],
    usgs: ['https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}',
           {maxZoom: 19, maxNativeZoom: 16, attribution: 'USGS The National Map'}],
    grey: ['https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
           {maxZoom: 19, maxNativeZoom: 16, attribution: 'Esri, HERE, Garmin, (c) OpenStreetMap contributors, and the GIS user community'}],
    topo: ['https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
           {maxZoom: 17, subdomains: 'abc', attribution: '© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)'}],
  };
  const tile = k => L.tileLayer(TILES[k][0], TILES[k][1]);
  // Project outlines in black and white, told apart by line; the measurement
  // layers (coherence, velocity, points) keep their colour scales.
  const STYLE = {
    bbox: {color: '#ffffff', weight: 2, dashArray: '8 6', fill: false},
    aoi: {color: '#ffffff', weight: 3, fillColor: '#ffffff', fillOpacity: 0.06},
    mine: {color: '#ffffff', weight: 2, fillColor: '#ffffff', fillOpacity: 0.28},
    report: {color: '#d0d0d0', weight: 2, fill: false, dashArray: '2 5'},
    frame: {color: '#9a9a9a', weight: 2, fill: false, dashArray: '14 6'},
    water: {color: '#bdbdbd', weight: 1, fillColor: '#bdbdbd', fillOpacity: 0.35},
    // a study's own layers (app_server.study_layers): pads, faults, fast zones, the stable area
    study_poly: {color: '#ffffff', weight: 2, fill: false, dashArray: '1 5', lineCap: 'round'},
    study_line: {color: '#ffffff', weight: 2.5, dashArray: '10 4 2 4'},
    fast_zone: {color: '#ffffff', weight: 2, fillColor: '#000000', fillOpacity: 0.35},
    stable: {color: '#bdbdbd', weight: 2, fill: false, dashArray: '4 4'},
  };
  const SWATCH = {bbox: '#8c8c8c', aoi: '#f2f2f2', mine: '#f2f2f2', report: '#bdbdbd', frame: '#8c8c8c',
                  raster: '#f2f2f2', points: '#f2f2f2', gnss: '#f2f2f2', water: '#bdbdbd', feature: '#f2f2f2',
                  study_point: '#f2f2f2', study_poly: '#f2f2f2', study_line: '#f2f2f2', fast_zone: '#6a6a6a', stable: '#bdbdbd'};

  // ------------------------------------------------------------ the mist
  // The one colour on the page: a small interferogram, 64 x 24 pixels, of two
  // subsiding point sources (Mogi: u = d / (r^2 + d^2)^1.5) on a slight tilt,
  // with speckle. Its wrapped phase is coloured round one cycle and washed out
  // toward white, then shown blown up without smoothing (--mist in app.css).
  // Decoration only: drawn from the formula, never from a project's data.
  function mist() {
    const W = 64, H = 24, c = document.createElement('canvas');
    c.width = W; c.height = H;
    const g = c.getContext('2d'), img = g.createImageData(W, H);
    const wheel = [[63, 102, 196], [47, 151, 190], [79, 179, 165], [147, 194, 94], [230, 195, 90], [224, 135, 77], [196, 90, 132]];
    const src = [[16, 14, 7, 3.2], [47, 9, 9, 2.4]];          // x, y, depth, fringes at the peak
    let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let ph = 0.012 * x + 0.02 * y;                           // the tilt
      for (const [sx, sy, d, n] of src) ph += n * Math.pow(d * d / ((x - sx) ** 2 + (y - sy) ** 2 + d * d), 1.5);
      ph += (rnd() - 0.5) * 0.12;                              // speckle
      const f = (ph - Math.floor(ph)) * wheel.length, k = Math.floor(f), t = f - k;
      const a = wheel[k], b = wheel[(k + 1) % wheel.length], i = (y * W + x) * 4;
      for (let j = 0; j < 3; j++) img.data[i + j] = Math.round(0.42 * (a[j] + (b[j] - a[j]) * t) + 0.58 * 242);  // the mist: 58% white
      img.data[i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    // blown up here, block by block, with a dark seam between pixels: a browser
    // smooths a background picture it scales, and the pixels are the point
    const S = 16, big = document.createElement('canvas');
    big.width = W * S; big.height = H * S;
    const bg = big.getContext('2d');
    bg.imageSmoothingEnabled = false;
    bg.drawImage(c, 0, 0, W * S, H * S);
    bg.fillStyle = 'rgba(0,0,0,0.09)';
    for (let x = 0; x < W; x++) bg.fillRect(x * S, 0, 1, H * S);
    for (let y = 0; y < H; y++) bg.fillRect(0, y * S, W * S, 1);
    document.documentElement.style.setProperty('--mist', `url(${big.toDataURL()})`);
  }
  mist();

  // ------------------------------------------------------------ helpers
  const STATE_WORD = {running: 'Running', done: 'Finished', failed: 'Failed', stopped: 'Stopped',
                      interrupted: 'Interrupted', idle: 'Not started', invalid: 'Config problem'};
  const pill = (st, words) => h('span', {class: 'pill ' + st}, words || STATE_WORD[st] || st);
  const pct = (d, t) => (t ? Math.max(0, Math.min(100, 100 * (d || 0) / t)) : 0);
  function bar(d, t, indet, live) {
    return h('div', {class: 'bar' + (indet ? ' indet' : live ? ' live' : '')}, h('i', {style: `width:${pct(d, t)}%`}));
  }
  function verdictBits(p) {
    const out = [];
    if (p.combined) out.push(h('span', {class: 'tag'}, 'COMBINED'));
    if (p.final === true) out.push(h('span', {class: 'tag final'}, 'FINAL'));
    else if (p.final === false) out.push(h('span', {class: 'tag draft'}, 'DRAFT'));
    if (p.send_rule) out.push(h('span', {class: p.send_rule.startsWith('hold') ? 'pill warn plain' : 'pill plain'}, p.send_rule));
    if (p.report_out_of_date || p.out_of_date) out.push(h('span', {class: 'pill warn plain'}, 'report out of date'));
    return out;
  }
  function areaRows(list) {
    return h('div', {class: 'areas'}, list.map(a => h('div', {class: 'area'},
      h('span', {class: 'n'}, a.name), h('span', {class: 'r'}, a.rate),
      a.range ? h('span', {class: 'rg'}, 'likely range ' + a.range.replace(/[()]/g, '')) : null,
      a.total ? h('span', {class: 'rg'}, a.total) : null,
      h('span', {class: 'rd ' + a.tone}, a.reading))));
  }
  const pad = n => String(n).padStart(3, '0');          // the scoreboard's digits
  const clock = () => new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});

  // ------------------------------------------------------------ router
  let route = {};
  let timers = [];
  function stopTimers() { timers.forEach(clearInterval); timers = []; }
  function every(ms, fn) { fn(); timers.push(setInterval(fn, ms)); }
  function go() {
    stopTimers();
    const hash = location.hash || '#/';
    const m = hash.match(/^#\/p\/(.+)$/), mf = hash.match(/^#\/files(?:\/(.+))?$/), mr = hash.match(/^#\/r\/(.+)$/);
    route = m ? {view: 'project', name: decodeURIComponent(m[1])}
      : mr ? {view: 'results', name: decodeURIComponent(mr[1])}
      : mf ? {view: 'files', name: mf[1] ? decodeURIComponent(mf[1]) : null}
      : hash === '#/new' ? {view: 'new'} : hash === '#/computer' ? {view: 'computer'} : {view: 'projects'};
    $$('.view').forEach(v => { v.hidden = v.id !== 'v-' + route.view; });
    $$('.navlink').forEach(a => a.classList.toggle('on', a.dataset.nav === (['project', 'results'].includes(route.view) ? 'projects' : route.view)));
    if (route.view !== 'results') $('#rframe').removeAttribute('src');   // the viewer stops when its section is left
    if (route.view === 'projects') showProjects();
    else if (route.view === 'results') showResults(route.name);
    else if (route.view === 'project') showProject(route.name);
    else if (route.view === 'computer') showComputer();
    else if (route.view === 'files') showFiles(route.name);
    else if (route.view === 'new') showNew();
    window.scrollTo(0, 0);
  }
  window.addEventListener('hashchange', go);

  // ============================================================ projects
  let homeMap = null, homeMarkers = null, projects = [], filter = 'all';
  const attention = p => ['failed', 'interrupted', 'invalid'].includes(p.state) ||
    (p.send_rule || '').startsWith('hold') || p.report_out_of_date;
  function showProjects() {
    if (!homeMap) {
      homeMap = L.map('homemap', {zoomControl: false, attributionControl: true, scrollWheelZoom: false}).setView([39, -96], 4);
      tile('osm').addTo(homeMap);
      homeMarkers = L.layerGroup().addTo(homeMap);
    }
    setTimeout(() => homeMap.invalidateSize(), 50);
    let fitted = false;
    every(5000, async () => {
      try {
        const d = await api('home');
        projects = d.projects;
        drawProjects();
        if (!fitted) { fitHome(); fitted = true; }
      } catch (e) { toast(e.message, 'bad'); }
    });
  }
  function fitHome() {
    const pts = projects.filter(p => p.center).map(p => p.center);
    if (pts.length === 1) homeMap.setView(pts[0], 9);
    else if (pts.length) homeMap.fitBounds(pts, {padding: [40, 40], maxZoom: 9});
  }
  function drawProjects() {
    const q = ($('#psearch').value || '').toLowerCase();
    const all = projects;
    const gone = all.filter(p => p.gone);
    drawGone(gone.filter(p => !q || p.name.toLowerCase().includes(q)));
    const projects_ = all.filter(p => !p.gone);
    const shown = projects_.filter(p => (!q || p.name.toLowerCase().includes(q)) && (
      filter === 'all' || (filter === 'running' && p.state === 'running') || (filter === 'attention' && attention(p)) ||
      (filter === 'done' && p.state === 'done'))).sort((a, b) => (b.state === 'running') - (a.state === 'running') || b.updated_t - a.updated_t);
    const nRun = projects_.filter(p => p.state === 'running').length;
    const nAttn = projects_.filter(attention).length;
    const nFinal = projects_.filter(p => p.final).length;
    put($('#kpis'), 
      h('div', {class: 'kpi'}, h('b', {}, pad(projects_.length)), h('span', {}, 'projects')),
      h('div', {class: 'kpi live'}, h('b', {}, pad(nRun)), h('span', {}, nRun === 1 ? 'run going now' : 'runs going now')),
      h('div', {class: 'kpi attn'}, h('b', {}, pad(nAttn)), h('span', {}, 'need attention')),
      h('div', {class: 'kpi'}, h('b', {}, pad(nFinal)), h('span', {}, 'signed off (final)')));
    const cards = shown.map(p => {
      const live = p.live;
      return h('a', {class: 'pcard card' + (p.state === 'running' ? ' is-running' : ''), href: '#/p/' + encodeURIComponent(p.name)},
        h('div', {class: 'top'}, h('div', {}, h('h3', {}, p.name), h('div', {class: 'route'}, p.route)), pill(p.state)),
        p.error ? h('div', {class: 'note bad'}, p.error) : null,
        live ? h('div', {class: 'livebar'}, h('div', {}, h('b', {}, live.label),
          live.total ? `  ${live.done}/${live.total}` : '', live.remaining ? ` · ${live.remaining} left` : ''),
          bar(live.done, live.total, !live.total, true)) : null,
        verdictBits(p).length ? h('div', {class: 'verdict'}, verdictBits(p)) : null,
        p.areas && p.areas.length ? areaRows(p.areas.slice(0, 3)) : (!live && !p.error ? h('div', {class: 'muted small'}, 'No results yet.') : null),
        h('div', {class: 'updated'}, p.updated ? 'Updated ' + p.updated : ''));
    });
    put($('#cards'), ...(cards.length ? cards : [h('div', {class: 'empty'},
      projects.length ? 'No project matches.' : 'No projects yet. Press “New project” to draw the first one.')]));
    homeMarkers.clearLayers();
    for (const p of projects) {
      if (!p.center || p.gone) continue;
      const col = p.state === 'running' || attention(p) ? '#ffffff' : p.state === 'done' ? '#bdbdbd' : '#6a6a6a';
      const b = p.bbox;
      L.rectangle([[b[1], b[0]], [b[3], b[2]]], {color: col, weight: 2, fillOpacity: 0.15}).addTo(homeMarkers);
      L.circleMarker(p.center, {radius: attention(p) ? 8 : 6, color: '#000', weight: attention(p) ? 4 : 2, fillColor: col, fillOpacity: 1})
        .bindTooltip(p.name + ' · ' + (STATE_WORD[p.state] || p.state))
        .on('click', () => { location.hash = '#/p/' + encodeURIComponent(p.name); }).addTo(homeMarkers);
    }
  }
  // Projects whose folder was deleted after their run (the experiments), and
  // configs that no longer load: listed apart, with what survives of them.
  function drawGone(list) {
    const box = $('#gone');
    box.hidden = !list.length;
    if (!list.length) return;
    $('#gonecount').textContent = `${list.length} · folder removed or config broken`;
    put($('#gonelist'), list.map(p => h('div', {class: 'gonerow'},
      h('div', {class: 'gn'}, h('b', {}, p.name), pill(p.gone === 'config' ? 'invalid' : 'idle', p.gone === 'config' ? 'config broken' : 'folder removed'),
        p.kept && p.kept.find(k => k.purpose) ? h('span', {class: 'muted small'}, p.kept.find(k => k.purpose).purpose) : null),
      p.gone === 'config' ? h('div', {class: 'muted small'}, p.error) : null,
      (p.kept || []).length ? h('ul', {class: 'kept'}, p.kept.map((k, i) => h('li', {},
        h('span', {}, (k.kind === 'results' ? 'Kept: ' : k.kind === 'archive' ? 'Archived: ' : '') + k.what + (k.finished ? ` · ${k.finished}` : '')
          + (k.status ? ` · ${k.status}` : '')),
        k.path ? h('button', {class: 'small', type: 'button', onclick: () => openKept(p, (p.kept || []).filter(x => x.path).indexOf(k))}, 'Open') : null)))
        : h('div', {class: 'muted small'}, 'Nothing of it is kept on this computer.'),
      h('div', {class: 'row'}, h('button', {class: 'small', type: 'button', onclick: () => forget(p)}, 'Forget'),
        h('span', {class: 'muted small'}, 'takes it off this list; the config file is kept')))));
  }
  async function openKept(p, i) {
    try { await api('open_kept', {config: p.config, index: i}); toast('Opening it…'); } catch (e) { toast(e.message, 'bad'); }
  }
  async function forget(p) {
    if (!await confirmBox(`Forget ${p.name}?`, 'It comes off the projects list. Nothing is deleted: the config file stays where it is, and the kept results stay too.', 'Forget')) return;
    try {
      const r = await api('forget', {config: p.config});
      toast(r.still_listed ? `${p.name} stays listed: its config sits in a configs folder the pipeline reads.` : `${p.name} is off the list.`, r.still_listed ? 'warn' : '');
      const d = await api('home'); projects = d.projects; drawProjects();
    } catch (e) { toast(e.message, 'bad'); }
  }
  $('#psearch').addEventListener('input', drawProjects);
  $$('#pfilter button').forEach(b => b.addEventListener('click', () => {
    filter = b.dataset.f;
    $$('#pfilter button').forEach(x => x.classList.toggle('on', x === b));
    drawProjects();
  }));
  $('#newproj').addEventListener('click', () => { location.hash = '#/new'; });

  // ============================================================ new
  async function showNew() {
    try {
      const d = await api('builder');
      $('#builderurl').textContent = d.url;
    } catch (e) { toast(e.message, 'bad'); }
  }
  $('#openbuilder').addEventListener('click', async () => {
    try {
      const d = await api('builder');
      window.open(d.url, '_blank');
      toast('The map builder opened in your browser. The project shows up under Projects once you create it.');
    } catch (e) { toast(e.message, 'bad'); }
  });

  // ============================================================ project
  let P = null;           // the project shown: {config, name, data}
  let pmap = null, base = null, mapLayers = {}, layerState = {}, fitted = false;
  let logState = {path: null, pos: null};
  let activity = [];

  async function showProject(name) {
    const d = await api('home').catch(e => { toast(e.message, 'bad'); return null; });
    const p = d && d.projects.find(x => x.name === name);
    if (!p) { $('#pname').textContent = name; put($('#pnotes'), h('div', {class: 'note bad'}, 'No such project here.')); return; }
    if (!P || P.config !== p.config) {
      P = {config: p.config, name};
      resetMap();
      logState = {path: null, pos: null};
      put($('#logtext'), );
      activity = [];
      put($('#activity'), );
    }
    $('#pname').textContent = name;
    initMap();
    every(2000, refreshProject);
    every(2500, refreshLog);
    every(4000, refreshResources);
  }

  async function refreshProject() {
    let d;
    try { d = await api('project', {config: P.config}); } catch (e) { put($('#pnotes'), h('div', {class: 'note bad'}, e.message)); return; }
    P.data = d;
    const hd = d.head;
    put($('#pmeta'), pill(hd.state, hd.state === 'running' && hd.state_words !== 'running' ? hd.state_words : null),
      h('span', {}, hd.route),
      hd.kind ? h('span', {}, `· ${hd.kind} run, started ${hd.started}${hd.took ? (hd.state === 'running' ? ', running ' : ', took ') + hd.took : ''}`) : null);
    const running = hd.state === 'running';
    $('#bstart').hidden = running;
    $('#bstop').hidden = !running;
    $('#breport').disabled = !(d.results && d.results.has_report);
    $('#bviewer').disabled = !(d.results && d.results.has_points);
    // Sign off: a report to sign that is not final yet; never during a run or another sign-off
    const so = d.results && d.results.signoff;
    $('#bsign').hidden = !(d.results && d.results.has_report) || d.results.final || hd.backend === 'combined';
    $('#bsign').disabled = running || !!(so && so.state === 'working');
    const was = P.signState;
    P.signState = so && so.state;
    if (was === 'working' && so && so.state !== 'working') toast(so.message, so.state === 'failed' ? 'bad' : so.package ? '' : 'warn');
    const link = name => h('a', {href: '#/p/' + encodeURIComponent(name)}, name);
    const extra = [];
    if (hd.sources) extra.push(h('div', {class: 'note info'}, 'Open: ',
      ...hd.sources.flatMap((x, i) => [i ? ' · ' : '', link(x.name), ` (${x.direction})`])));
    if (hd.combine && hd.combine.kind === 'combined') extra.push(h('div', {class: 'note ' + (hd.combine.status === 'current' ? 'info' : 'warn')},
      'Also measured from the other pass: combined in ', link(hd.combine.name), hd.combine.status === 'current' ? '.' : ` (${hd.combine.status}).`));
    if (hd.combine && hd.combine.kind === 'offer') extra.push(h('div', {class: 'note info'},
      h('button', {type: 'button', class: 'primary', onclick: () => combineWith(hd.combine)}, 'Combine with ' + hd.combine.name)));
    put($('#pnotes'), ...hd.notes.map(n => h('div', {class: 'note ' + n.tone}, n.text)), ...extra);
    // a combined project is one step (Combine): no stepper, its own words in the stages tab
    $('#phases').hidden = hd.backend === 'combined';
    drawPhases(d.phases);
    drawNow(d.stages, running);
    if (hd.backend === 'combined') put($('#t-stages'), h('p', {class: 'muted', style: 'padding:16px'},
      d.results ? `Combined ${d.results.generated.replace('T', ' ')}. One step: Start, then Combine, builds it again.`
        : 'Not combined yet: press Start, then Combine.'));
    else drawStages(d.stages);
    drawResults(d.results);
    syncLayers(d.layers);
  }

  // What each step does, as a 12 x 12 sprite: '#' a pixel, '+' a pixel that
  // blinks while the step works. The satellite and its beam; three images
  // sliding into line; the network of pairs phase linking reads; wrapped phase
  // (a sawtooth) above the ramp it unwraps to; a point's series falling over
  // time; the report.
  const SPRITE = {
    data: ['....####....', '.##.#..#.##.', '.##.####.##.', '.....##.....', '.....++.....', '....+..+....',
           '...+....+...', '..+......+..', '.+........+.', '+..........+', '############', '............'],
    stack: ['.#########..', '.#.......#..', '.#########..', '............', '..+++++++++.', '..+.......+.',
            '..+++++++++.', '............', '.#########..', '.#.......#..', '.#########..', '............'],
    link: ['##........##', '##+......+##', '..+......+..', '...+....+...', '....+..+....', '.....##.....',
           '.....##.....', '....+..+....', '...+....+...', '##+......+##', '##++++++++##', '............'],
    unwrap: ['#...#...#...', '##..##..##..', '#.#.#.#.#.#.', '#..##..##..#', '#...#...#...', '............',
             '...........+', '.........++.', '.......++...', '.....++.....', '...++.......', '.++.........'],
    series: ['#...........', '#.##........', '#...........', '#....++.....', '#...........', '#.......##..',
             '#...........', '#..........#', '#...........', '#...........', '############', '............'],
    products: ['.#######....', '.#.....##...', '.#.....#.#..', '.#.....####.', '.#........#.', '.#.++++++.#.',
               '.#........#.', '.#.++++++.#.', '.#........#.', '.#.+++....#.', '.#........#.', '.##########.'],
  };
  const BADGE = {'✓': ['.....', '....#', '...#.', '#.#..', '.#...'], '✕': ['#...#', '.#.#.', '..#..', '.#.#.', '#...#'],
                 '!': ['..#..', '..#..', '..#..', '.....', '..#..']};
  function sprite(rows) {
    const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    s.setAttribute('viewBox', `0 0 ${rows[0].length} ${rows.length}`);
    rows.forEach((row, y) => [...row].forEach((c, x) => {
      if (c === '.') return;
      const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      r.setAttribute('x', x); r.setAttribute('y', y); r.setAttribute('width', 1); r.setAttribute('height', 1);
      if (c === '+') r.setAttribute('class', 'b');
      s.append(r);
    }));
    return s;
  }
  const glyph = key => sprite(SPRITE[key] || ['.']);
  function drawPhases(ph) {
    const word = {done: 'Done', earlier: 'Done in an earlier run', active: 'Working…', failed: 'Failed', stopped: 'Stopped part-way',
                  skipped: 'Not needed this run', waiting: 'Up next', 'not run': 'Not run yet'};
    const icon = {done: '✓', earlier: '✓', failed: '✕', stopped: '!'};
    put($('#phases'), ...ph.map((p, i) => h('li', {class: 'phase ' + p.status.replace(' ', '-')},
      h('div', {class: 'ic', title: p.label}, glyph(p.key), icon[p.status] ? h('span', {class: 'badge'}, sprite(BADGE[icon[p.status]])) : null),
      h('div', {class: 'nm'}, p.label),
      h('div', {class: 'st'}, word[p.status] || p.status, p.background ? ' · a download still going in the background' : ''),
      p.status === 'active' && p.fraction != null ? bar(p.fraction, 1, false, true) : null)));
  }

  function drawNow(stages, running) {
    const live = stages.filter(s => s.live);
    if (!running || !live.length) { put($('#now'), ); return; }
    put($('#now'), ...live.map(s => h('div', {class: 'nowcard card'},
      h('div', {class: 'l'}, h('b', {}, s.label), s.total ? h('span', {class: 'big'}, Math.floor(pct(s.done, s.total)) + '%') : h('span', {class: 'muted'}, 'working…')),
      bar(s.done, s.total, !s.total, true),
      h('div', {class: 'sub'},
        s.total ? h('span', {}, `${s.done} of ${s.total}`) : null,
        h('span', {}, 'elapsed ' + s.elapsed),
        s.remaining ? h('span', {}, h('b', {}, s.remaining), ' left') : null,
        s.note ? h('span', {}, s.note.replace(/^log: .*\//, 'log: ')) : null))));
  }

  const MARK = {'✓': 'OK', '✗': 'XX', '!': '!!', '■': '||', '–': '--', '·': '..'};
  const PH_NAMES = {data: 'Get the data', stack: 'Line up the images', link: 'Phase linking', unwrap: 'Unwrap',
                    series: 'Time series', products: 'Products & report', null: 'Run'};
  function drawStages(stages) {
    const box = $('#t-stages');
    if (!stages.length) { put(box, h('p', {class: 'muted', style: 'padding:16px'}, 'Nothing has run here yet. Press Start.')); return; }
    const groups = [];
    for (const s of stages) {
      const g = groups.find(x => x.k === s.phase) || (groups.push({k: s.phase, items: []}), groups[groups.length - 1]);
      g.items.push(s);
    }
    const top = box.scrollTop;
    put(box, ...groups.map(g => h('div', {class: 'stagegroup'}, h('h4', {}, PH_NAMES[g.k] || g.k),
      g.items.map(s => h('div', {class: 'stage ' + s.tone, title: s.note || ''},
        h('span', {class: 'si'}, s.live ? '>>' : MARK[s.icon] || s.icon),
        h('span', {}, s.label),
        h('span', {class: 'sm'}, s.total ? `${s.done}/${s.total}` : '', s.elapsed ? '  ' + s.elapsed : ''),
        s.live && s.total ? bar(s.done, s.total, false, true) : null,
        s.note ? h('span', {class: 'sn'}, s.note) : null)))));
    box.scrollTop = top;
  }

  function drawResults(r) {
    const box = $('#t-results');
    if (!r) { put(box, h('div', {class: 'results'}, h('p', {class: 'muted'}, 'Results appear here when the client report is built (the last phase).'))); return; }
    put(box, h('div', {class: 'results'},
      h('div', {class: 'verdict'}, verdictBits(r)),
      r.final ? h('div', {class: 'note ok'}, `FINAL, reviewed by ${r.reviewed_by || '?'}.`)
        : h('div', {class: 'note warn'}, h('b', {}, `DRAFT: ${r.blockers.length} thing(s) to settle before it goes to a client.`),
          r.blockers.length ? h('ul', {}, r.blockers.slice(0, 5).map(b => h('li', {}, b))) : null,
          h('div', {class: 'small muted'}, 'Read the whole report, then Sign off (beside Report): it rebuilds the report and builds the client package once nothing else blocks it.')),
      signoffNote(r.signoff),
      h('div', {class: 'muted small'}, r.frame),
      areaRows(r.areas),
      r.features.map(f => h('div', {class: 'small ' + (f.tone === 'warn' ? '' : 'muted')}, f.text)),
      h('div', {class: 'muted small'}, 'Report generated ' + r.generated)));
  }

  // what the sign-off going on (or just finished) in this app has done
  function signoffNote(j) {
    if (!j) return null;
    if (j.state === 'working') return h('div', {class: 'note info'}, 'Signing off: ' + (j.steps[j.steps.length - 1] || 'recording the sign-off…'));
    return h('div', {class: 'note ' + (j.state === 'failed' ? 'bad' : j.package ? 'ok' : 'warn')}, j.message);
  }

  // ------------------------------------------------------------ the live map
  function initMap() {
    if (pmap) { setTimeout(() => pmap.invalidateSize(), 50); return; }
    pmap = L.map('pmap', {preferCanvas: true});
    base = tile('imagery').addTo(pmap);
    pmap.setView([39, -96], 4);
    setTimeout(() => pmap.invalidateSize(), 50);
  }
  function resetMap() {
    if (pmap) Object.values(mapLayers).forEach(l => pmap.removeLayer(l));
    mapLayers = {}; layerState = {}; fitted = false;
    $('#legend').hidden = true;
    put($('#layers'), );
  }
  $$('#basemaps button').forEach(b => b.addEventListener('click', () => {
    if (!pmap) return;
    pmap.removeLayer(base);
    base = tile(b.dataset.b).addTo(pmap);
    base.bringToBack();
    $$('#basemaps button').forEach(x => x.classList.toggle('on', x === b));
  }));

  function logActivity(text) {
    activity.unshift({t: clock(), text});
    put($('#activity'), ...activity.slice(0, 200).map(a => h('li', {}, h('time', {}, a.t), h('span', {}, a.text))));
  }

  // Layers that start switched on; the velocity picture waits for a click so
  // the quality picture is not hidden under it.
  const ON_BY_DEFAULT = new Set(['bbox', 'aoi', 'underground_mines', 'surface_mines', 'other_movement', 'frame', 'vertical', 'gnss', 'features', 'combined_vertical']);
  async function syncLayers(list) {
    const first = !Object.keys(layerState).length;
    // before the products exist, the quality picture is the one to watch
    const hasVert = list.some(x => x.id === 'vertical' && x.ready);
    for (const L0 of list) {
      const st = layerState[L0.id] || (layerState[L0.id] = {on: ON_BY_DEFAULT.has(L0.id) || L0.id.startsWith('report_') || L0.id.startsWith('study_') ||
        (!hasVert && L0.id === 'coherence'), since: null, ready: false});
      st.meta = L0;
      const changed = L0.ready && (st.since !== L0.since || !st.ready);
      if (changed) {
        const wasReady = st.ready;
        st.ready = true; st.since = L0.since; st.fresh = !first;
        if (!first) logActivity((wasReady ? 'Updated on the map: ' : 'New on the map: ') + L0.label);
        if (st.on) await loadLayer(L0.id).catch(e => { st.error = e.message; });
        else { if (mapLayers[L0.id]) { pmap.removeLayer(mapLayers[L0.id]); delete mapLayers[L0.id]; } }
      }
      if (!L0.ready) st.ready = false;
    }
    drawLayerList(list);
    if (!fitted && mapLayers.bbox) { pmap.fitBounds(mapLayers.bbox.getBounds(), {padding: [20, 20]}); fitted = true; }
  }

  async function loadLayer(id) {
    const d = await api('layer', {config: P.config, id});
    if (mapLayers[id]) pmap.removeLayer(mapLayers[id]);
    const meta = (layerState[id] || {}).meta || {};
    let layer;
    if (d.kind === 'geojson') {
      layer = L.geoJSON(d.data, {style: STYLE[meta.style] || STYLE.report, interactive: true,
        onEachFeature: (f, l) => { const n = (f.properties || {}).name; l.bindTooltip(n ? `${meta.label}: ${n}` : meta.label); }});
    } else if (d.kind === 'image') {
      layer = L.imageOverlay('data:image/png;base64,' + d.png, d.bounds, {opacity: 0.85, interactive: false});
      layerState[id].legend = d.legend;
      layerState[id].source = `${d.source}, made ${d.made}`;
    } else if (d.kind === 'points') {
      const g = L.layerGroup();
      const rend = L.canvas({padding: 0.3});
      for (let i = 0; i < d.lat.length; i++) {
        const t = Math.max(-1, Math.min(1, d.v[i] / d.limit));
        L.circleMarker([d.lat[i], d.lon[i]], {renderer: rend, radius: 2.6, stroke: false, fillOpacity: 0.9, fillColor: diverging(t)}).addTo(g);
      }
      layer = g;
      layerState[id].legend = {min: -d.limit, max: d.limit, units: 'in/yr' + (d.relative ? ' (relative)' : ''), colors: [-1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1].map(diverging)};
      layerState[id].source = `${d.shown.toLocaleString()} of ${d.n.toLocaleString()} points shown; the results map has them all`;
      if (d.stale) logActivity(d.stale);
    } else if (d.kind === 'features') {
      const g = L.layerGroup();
      for (const f of d.features) {
        L.marker([f.lat, f.lon], {icon: L.divIcon({className: 'featmark', html: String(f.n), iconSize: [22, 22]})})
          .bindTooltip(f.text).addTo(g);
      }
      layer = g;
    } else if (d.kind === 'study_points') {
      // a study's points (wells), coloured by the report's reading of each
      const g = L.layerGroup();
      const col = Object.fromEntries((d.legend.classes || []).map(c => [c.name, c.color]));
      for (const p of d.points) {
        const fill = col[p.status_words] || '#bdbdbd';
        L.circleMarker([p.lat, p.lon], {radius: 6, color: fill === '#000000' ? '#ffffff' : '#000000', weight: 2,
                                        fillColor: fill, fillOpacity: p.status === 'no_report' ? 0.4 : 1})
          .bindTooltip(`${meta.label}: ${p.id}`).bindPopup(() => wellPopup(p)).addTo(g);
      }
      layer = g;
      layerState[id].legend = d.legend;
      layerState[id].source = d.from_report ? `${d.points.length} from the report` : `${d.points.length} listed; numbers after Products & report`;
    } else if (d.kind === 'gnss') {
      const g = L.layerGroup();
      for (const s of d.stations) {
        L.circleMarker([s.lat, s.lon], {radius: 7, color: '#000', weight: 3, fillColor: s.used ? '#ffffff' : '#6a6a6a', fillOpacity: 1})
          .bindTooltip(`${s.site}${s.used ? ' (used in the tie)' : ' (not used)'} · ${s.km} km`).addTo(g);
      }
      layer = g;
    }
    mapLayers[id] = layer.addTo(pmap);
    if (layerState[id] && layerState[id].legend) { legendOrder = legendOrder.filter(x => x !== id); legendOrder.push(id); }
    if (mapLayers.bbox && id !== 'bbox') mapLayers.bbox.bringToFront && mapLayers.bbox.bringToFront();
    showLegend();
  }
  // a well's popup: id, its own columns, the reading, value and 95% range, shared values, tier
  function wellPopup(p) {
    const n = v => (v == null ? '' : (v > 0 ? '+' : '') + Number(v).toFixed(2));
    const attrs = Object.entries(p.attributes || {}).filter(([, v]) => v != null && v !== '');
    return h('div', {class: 'wellpop'},
      h('b', {}, p.id),
      attrs.length ? h('div', {}, attrs.map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`).join(' · ')) : null,
      h('div', {}, p.status_words),
      p.value_in_yr != null ? h('div', {}, `${n(p.value_in_yr)} in/yr vertical, relative to ordinary ground` +
        (p.range_in_yr ? ` (95% range ${n(p.range_in_yr[0])} to ${n(p.range_in_yr[1])})` : '')) : null,
      p.reason ? h('div', {class: 'muted'}, p.reason) : null,
      (p.shared_with || []).length ? h('div', {class: 'muted'}, `Shared value with ${p.shared_with.join(', ')}: not a per-well rate`) : null,
      p.tier ? h('div', {class: 'muted'}, `Tier: ${p.tier}`) : null);
  }
  function diverging(t) {   // -1 red (sinking / away) .. 0 white .. +1 blue
    const a = [178, 24, 43], m = [247, 247, 247], b = [33, 102, 172];
    const c = t < 0 ? a.map((x, i) => m[i] + (x - m[i]) * -t) : b.map((x, i) => m[i] + (x - m[i]) * t);
    return `rgb(${c.map(Math.round).join(',')})`;
  }
  let legendOrder = [];
  function showLegend() {
    const id = legendOrder.filter(x => mapLayers[x] && layerState[x] && layerState[x].legend).pop();
    if (!id) { $('#legend').hidden = true; return; }
    const lg = layerState[id].legend, meta = layerState[id].meta;
    $('#legend').hidden = false;
    if (lg.classes) {
      put($('#legend'), h('b', {}, meta.label), h('div', {class: 'classes'}, lg.classes.map(c =>
        h('div', {}, h('i', {style: `background:${c.color}`}), c.name))));
      return;
    }
    const f = v => (Math.abs(lg.max) < 2 ? v.toFixed(2) : v.toFixed(1));
    put($('#legend'), h('b', {}, meta.label),
      h('div', {class: 'ramp', style: `background:linear-gradient(90deg,${lg.colors.join(',')})`}),
      h('div', {class: 'ends'}, h('span', {}, f(lg.min)), h('span', {}, lg.units || ''), h('span', {}, f(lg.max))),
      lg.style === 'div' ? h('div', {class: 'muted'}, id.startsWith('los') ? 'red: moving away from the satellite' : 'red: going down') : null,
      lg.style === 'div_ew' ? h('div', {class: 'muted'}, 'red: moving east, blue: moving west (indicative)') : null,
      h('div', {class: 'muted'}, 'a look: the figures are the report’s'));
  }
  function drawLayerList(list) {
    const groups = {};
    for (const L0 of list) (groups[L0.group] = groups[L0.group] || []).push(L0);
    const items = [];
    let lastGroup = null;
    for (const L0 of list) {
      if (L0.group !== lastGroup) { items.push(h('li', {class: 'lgroup'}, L0.group)); lastGroup = L0.group; }
      const st = layerState[L0.id];
      const sw = h('span', {class: 'sw', style: `border-color:${SWATCH[L0.style]};background:${SWATCH[L0.style]}33`});
      if (!L0.ready) {
        items.push(h('li', {class: 'pending'}, sw, h('span', {class: 'lt'}, L0.label, h('small', {}, L0.note || 'appears after: ' + L0.after))));
        continue;
      }
      const cb = h('input', {type: 'checkbox', checked: st.on, 'aria-label': L0.label});
      const li = h('li', {class: 'ready' + (st.fresh ? ' fresh' : ''), title: L0.about || ''}, cb, sw,
        h('span', {class: 'lt'}, L0.label, h('small', {}, st.error ? 'could not draw: ' + st.error : st.source || (L0.since ? 'since ' + L0.since : ''))));
      st.fresh = false;
      const toggle = async () => {
        st.on = !st.on; cb.checked = st.on;
        if (st.on) { try { await loadLayer(L0.id); st.error = null; } catch (e) { st.error = e.message; toast(e.message, 'warn'); } }
        else if (mapLayers[L0.id]) { pmap.removeLayer(mapLayers[L0.id]); delete mapLayers[L0.id]; showLegend(); }
        if (st.on) showLegend();
      };
      li.addEventListener('click', e => { if (e.target !== cb) toggle(); });
      cb.addEventListener('change', () => { st.on = !cb.checked; toggle(); });
      items.push(li);
    }
    put($('#layers'), ...items);
    const n = list.filter(x => x.ready).length;
    $('#mapsub').textContent = `${n} of ${list.length} layers ready · more appear as the run makes them`;
  }

  // ------------------------------------------------------------ log, resources
  async function refreshLog() {
    if (!P) return;
    let d;
    try { d = await api('log', {config: P.config, path: logState.path, pos: logState.pos}); } catch (e) { return; }
    const pre = $('#logtext');
    if (d.reset) put(pre, );
    logState = {path: d.path, pos: d.pos};
    $('#logpath').textContent = d.path ? (d.sub ? 'the running step’s own log: ' : '') + d.path : 'no log yet';
    if (!d.lines.length) return;
    const frag = document.createDocumentFragment();
    for (const line of d.lines) {
      const cls = /\b(ERROR|CRITICAL|Traceback)\b/.test(line) ? 'e' : /\bWARNING\b/.test(line) ? 'w' : null;
      frag.append(cls ? h('span', {class: cls}, line + '\n') : document.createTextNode(line + '\n'));
    }
    pre.append(frag);
    while (pre.childNodes.length > 4000) pre.firstChild.remove();
    if ($('#logfollow').checked) pre.scrollTop = pre.scrollHeight;
  }

  function resourceBlocks(r) {
    const out = [];
    if (r.budget_gb != null) {
      const tone = r.budget_gb > 150 ? 'ok' : r.budget_gb > 60 ? 'warn' : 'bad';
      out.push(h('div', {class: 'm'}, h('b', {style: `color:var(--${tone})`}, `${Math.round(r.budget_gb)} GB`),
        h('span', {}, 'disk the runs can use'),
        r.c_free_gb != null ? h('div', {class: 'muted small'}, `C: ${Math.round(r.c_free_gb)} GB free` + (r.vhdx_headroom_gb != null ? ` + ${Math.round(r.vhdx_headroom_gb)} GB unused in the WSL disk` : '')) : null));
    } else if (r.free_gb != null) {
      out.push(h('div', {class: 'm'}, h('b', {}, `${Math.round(r.free_gb)} GB`), h('span', {}, 'free disk')));
    }
    if (r.mem_total_gb) out.push(h('div', {class: 'm'}, h('b', {}, `${r.mem_used_gb.toFixed(1)} / ${Math.round(r.mem_total_gb)} GB`), h('span', {}, 'memory in use'), bar(r.mem_used_gb, r.mem_total_gb)));
    if (r.load != null) out.push(h('div', {class: 'm'}, h('b', {}, r.load.toFixed(1)), h('span', {}, `load on ${r.cpus} cores`), bar(r.load, r.cpus)));
    if (r.net_hist) {
      const hist = r.net_hist, mx = Math.max(1, ...hist);
      const dpath = hist.map((v, i) => `${i ? 'L' : 'M'}${(i / (hist.length - 1) * 100).toFixed(1)},${(28 - v / mx * 26).toFixed(1)}`).join(' ');
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', '0 0 100 28'); svg.setAttribute('preserveAspectRatio', 'none'); svg.classList.add('spark');
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'path'); p.setAttribute('d', dpath); p.setAttribute('vector-effect', 'non-scaling-stroke');
      svg.append(p);
      out.push(h('div', {class: 'm'}, h('b', {}, `${(r.net_mb_s || 0).toFixed(1)} MB/s`), h('span', {}, 'downloading'), svg));
    }
    return out;
  }
  async function refreshResources() {
    try {
      const r = await api('resources', {config: P && P.config});
      put($('#pres'), ...resourceBlocks(r));
    } catch (e) { /* the next tick */ }
  }

  // ------------------------------------------------------------ tabs and buttons
  $$('.tabs button').forEach(b => b.addEventListener('click', () => {
    $$('.tabs button').forEach(x => x.classList.toggle('on', x === b));
    $$('.side .tab').forEach(t => { t.hidden = t.id !== 't-' + b.dataset.t; });
    if (b.dataset.t === 'log' && $('#logfollow').checked) $('#logtext').scrollTop = 1e9;
  }));
  $('#bstop').addEventListener('click', async () => {
    if (!await confirmBox('Stop the run?', 'The run tears down what it started and leaves its work resumable: starting again picks up where it stopped.', 'Stop it')) return;
    try { const r = await api('stop', {config: P.config}); toast('Run ' + r.result + '.'); refreshProject(); } catch (e) { toast(e.message, 'bad'); }
  });
  $('#breport').addEventListener('click', async () => {
    const r = P.data && P.data.results;
    if (!r) return;
    if (!r.final || r.hold || r.out_of_date) {
      const why = (r.blockers || []).slice(0, 8).map(b => '• ' + b).join('\n');
      const text = 'This client report is a DRAFT: it is not ready to go to a client.\n\n' + why +
        (r.send_rule && (r.send_rule !== 'send' || r.out_of_date) ? `\n\nThe quality assessment says “${r.send_rule}”.` : '') +
        (r.out_of_date ? '\n\nThe report is older than the quality assessment: rebuild it.' : '') + '\n\nOpen it for internal review?';
      if (!await confirmBox('Open a draft report?', text, 'Open the draft')) return;
    }
    try { await api('open', {config: P.config, what: 'report'}); toast('Opening the report…'); } catch (e) { toast(e.message, 'bad'); }
  });
  $('#bviewer').addEventListener('click', () => { location.hash = '#/r/' + encodeURIComponent(route.name); });

  // ============================================================ results map (the viewer, inside the app)
  // The results viewer (results_viewer.py, its own server) in a frame: its CSP
  // lets only this app's page frame it. The app's theme goes with it in the
  // link, and again by message when B/W is pressed (viewer_app.js listens).
  let resURL = '';
  const appTheme = () => document.documentElement.dataset.theme ||
    (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  async function viewerURL(name) {
    const d = await api('viewer', {});
    return d.url.split('#')[0] + '#project=' + encodeURIComponent(name) + '&theme=' + appTheme();
  }
  async function showResults(name) {
    $('#rname').textContent = 'Results map · ' + name;
    $('#rback').href = '#/p/' + encodeURIComponent(name);
    try { resURL = await viewerURL(name); $('#rframe').src = resURL; } catch (e) { toast(e.message, 'bad'); }
  }
  $('#rreload').addEventListener('click', () => {
    if (!resURL) return;
    $('#rframe').src = 'about:blank';
    setTimeout(() => { $('#rframe').src = resURL; }, 30);
  });
  $('#rwindow').addEventListener('click', async () => {
    try { window.open(await viewerURL(route.name), '_blank'); } catch (e) { toast(e.message, 'bad'); }
  });

  // ------------------------------------------------------------ sign-off dialog
  // The terminal dashboard's Sign off (r): the same questions, and the server
  // runs the same functions (app_server.api_signoff). Checked here first so a
  // missing name or tick is said at once; the server refuses them anyway.
  let signFeats = [];
  $('#bsign').addEventListener('click', async () => {
    let f;
    try { f = await api('signoff_form', {config: P.config}); } catch (e) { toast(e.message, 'bad'); return; }
    if (f.running) { toast('Wait for the run to finish before signing off.', 'warn'); return; }
    if (f.job && f.job.state === 'working') { toast('A sign-off is already going on this project.', 'warn'); return; }
    signFeats = f.features;
    $('#sgname').value = f.reviewer || '';
    $('#sgspec').value = '';
    $('#sgnotes').value = '';
    $('#sghold').hidden = !f.held;
    $('#sgholdtext').textContent = f.held || '';
    put($('#sgleft'), f.other_blockers.length ? h('div', {class: 'note warn'}, h('b', {}, 'Still blocking a final release whatever you sign:'),
      h('ul', {}, f.other_blockers.slice(0, 8).map(b => h('li', {}, b)))) : null);
    $('#sgfeatwhat').textContent = f.features.length ? 'Tick each feature once you have checked it (imagery, dates, the ground):'
      : 'The report shows no features to check.';
    put($('#sgfeats'), f.features.map(x => h('label', {}, h('input', {type: 'checkbox', 'data-n': x.n}), h('span', {}, x.label))));
    put($('#sgerr'));
    $('#sggo').disabled = false;
    $('#dsign').showModal();
  });
  $('#sggo').addEventListener('click', async () => {
    const err = m => put($('#sgerr'), h('div', {class: 'note bad'}, m));
    const reviewer = $('#sgname').value.trim();
    const ticked = $$('#sgfeats input').filter(c => c.checked).map(c => Number(c.dataset.n));
    const held = !$('#sghold').hidden;
    const specialist = held ? $('#sgspec').value.trim() : '';
    if (!reviewer) return err("Type the reviewer's name.");
    if (ticked.length !== signFeats.length) return err(`Every feature must be checked before signing off (${ticked.length} of ${signFeats.length} ticked).`);
    if (held && !specialist) return err("This report is held for a specialist: type the specialist's name.");
    $('#sggo').disabled = true;
    try {
      await api('signoff', {config: P.config, reviewer, features: ticked, specialist, notes: $('#sgnotes').value.trim()});
      $('#dsign').close();
      toast(`Signed off by ${reviewer}; rebuilding the report…`);
      refreshProject();
    } catch (e) { err(e.message); $('#sggo').disabled = false; }
  });

  // ------------------------------------------------------------ start dialog
  let chosen = null;
  $('#bstart').addEventListener('click', openStart);
  async function openStart() {
    const d = $('#dstart');
    $('#sstep1').hidden = false; $('#sstep2').hidden = true;
    ['#sback', '#srecheck', '#sgo'].forEach(s => { $(s).hidden = true; });
    put($('#soptions'), h('p', {class: 'muted'}, 'Working out what fits the files on disk…'));
    $('#sreason').hidden = true;
    d.showModal();
    let o;
    try { o = await api('run_options', {config: P.config}); } catch (e) { put($('#soptions'), h('div', {class: 'note bad'}, e.message)); return; }
    if (o.running) { put($('#soptions'), h('div', {class: 'note warn'}, 'A run is already going here.')); return; }
    const rec = o.options.filter(x => x.recommended), rest = o.options.filter(x => !x.recommended);
    const btn = x => h('button', {type: 'button', class: 'opt' + (x.recommended ? ' rec' : ''), onclick: () => pick(x)},
      h('b', {}, (x.recommended ? 'Recommended: ' : '') + x.title),
      x.detail ? h('span', {}, x.detail) : null,
      x.recommended && o.reason ? h('span', {class: 'why'}, o.reason) : null,
      x.warning ? h('span', {class: 'wn'}, '! ' + x.warning) : null,
      h('span', {}, x.preflight ? 'The checks run first.' : 'Starts straight away.'));
    put($('#soptions'), ...rec.map(btn),
      rec.length ? h('div', {class: 'advanced'}, 'Advanced: only when you know why the recommendation does not fit') : null,
      ...rest.map(btn));
  }
  async function pick(x) {
    chosen = x;
    if (!x.preflight) {
      if (!await confirmBox('Start: ' + x.title + '?', x.detail || '', 'Start')) return;
      return start();
    }
    $('#sstep1').hidden = true; $('#sstep2').hidden = false;
    $('#sback').hidden = false; $('#srecheck').hidden = false; $('#sgo').hidden = false;
    $('#pfkind').textContent = x.title;
    $('#pfcleaned').hidden = !x.warning;
    $('#pfcleanedtext').textContent = 'I mean the whole pipeline: ' + x.warning;
    $('#pfcleanedbox').checked = false;
    runChecks();
  }
  async function runChecks() {
    $('#sgo').disabled = true;
    $('#pfstate').className = 'pill running'; $('#pfstate').textContent = 'checking';
    put($('#pfchecks'), h('li', {class: 'info'}, h('span', {class: 'ci'}, h('span', {class: 'spin'})),
      h('span', {}, 'Checking the logins, disk, memory and the plan. This can take a minute…')));
    $('#pfplan').textContent = '';
    let r;
    try { r = await api('preflight', {config: P.config, kind: chosen.kind}); } catch (e) {
      put($('#pfchecks'), h('li', {class: 'fail'}, h('span', {class: 'ci'}, 'XX'), h('span', {}, e.message)));
      $('#pfstate').className = 'pill bad'; $('#pfstate').textContent = 'could not check';
      return;
    }
    const mark = {ok: 'OK', warn: '!!', info: 'i', fail: 'XX'};
    put($('#pfchecks'), ...r.checks.map(c => h('li', {class: c.level}, h('span', {class: 'ci'}, mark[c.level] || '·'), h('span', {}, c.text))));
    $('#pfplan').textContent = r.plan.join('\n');
    $('#pfplandet').hidden = !r.plan.length;
    $('#pfstate').className = 'pill ' + (r.ok ? 'ok' : 'bad');
    $('#pfstate').textContent = r.ok ? 'ready' : 'fix XX first';
    const gate = () => { $('#sgo').disabled = !r.ok || (!!chosen.warning && !$('#pfcleanedbox').checked); };
    $('#pfcleanedbox').onchange = gate;
    gate();
  }
  $('#sback').addEventListener('click', openStart);
  $('#srecheck').addEventListener('click', runChecks);
  $('#sgo').addEventListener('click', start);
  async function combineWith(c) {
    if (!await confirmBox('Combine with ' + c.name + '?', 'This makes a new combined project from the two. Neither '
      + 'project is changed. It takes a minute or two.', 'Combine')) return;
    toast('Combining… this takes a minute or two.');
    try {
      const r = await api('combine_with', {config: P.config, partner: c.config});
      toast('Combined: ' + r.name);
      location.hash = '#/p/' + encodeURIComponent(r.name);
    } catch (e) { toast(e.message, 'bad'); }
  }
  async function start() {
    if (chosen.kind === 'combine') toast('Combining… this takes a minute or two.');
    try {
      const r = await api('start', {config: P.config, kind: chosen.kind, confirm_cleaned: $('#pfcleanedbox').checked});
      $('#dstart').close();
      if (r.finished) { toast('Done: ' + chosen.title + '.'); logActivity('Done: ' + chosen.title); refreshProject(); return; }
      toast(`Started: ${chosen.title}. It keeps going if you close this window.`);
      logActivity('Started: ' + chosen.title);
      refreshProject();
    } catch (e) { toast(e.message, 'bad'); }
  }

  // ============================================================ files
  const GBf = n => n == null ? '—' : n >= 1e9 ? (n / 1e9).toFixed(1) + ' GB' : n >= 1e6 ? (n / 1e6).toFixed(0) + ' MB'
    : n >= 1e3 ? (n / 1e3).toFixed(0) + ' KB' : n + ' B';
  let F = {config: null, name: null, rel: '', chosen: new Set(), list: null};
  async function showFiles(name) {
    let d;
    try { d = await api('home'); } catch (e) { toast(e.message, 'bad'); return; }
    const ok = d.projects.filter(p => !p.error);
    put($('#fprojects'), ok.map(p => h('li', {class: p.name === name ? 'on' : ''},
      h('a', {href: '#/files/' + encodeURIComponent(p.name)}, h('span', {}, p.name), pill(p.state)))));
    const p = ok.find(x => x.name === name);
    $('#fpanel').hidden = !p;
    if (!p) return;
    if (F.config !== p.config) F = {config: p.config, name: p.name, rel: '', chosen: new Set(), list: null};
    $('#fname').textContent = p.name;
    loadDir(F.rel);
  }
  async function loadDir(rel) {
    F.rel = rel; F.chosen.clear(); updateSel();
    put($('#ftable'), h('p', {class: 'muted', style: 'padding:16px'}, h('span', {class: 'wheel'}), '  measuring…'));
    let d;
    try { d = await api('files', {config: F.config, rel}); } catch (e) { put($('#ftable'), h('div', {class: 'note bad'}, e.message)); return; }
    F.list = d;
    const parts = d.rel ? d.rel.split('/') : [];
    put($('#fcrumbs'), h('a', {href: '#', onclick: e => { e.preventDefault(); loadDir(''); }}, F.name),
      parts.map((x, i) => [' / ', h('a', {href: '#', onclick: e => { e.preventDefault(); loadDir(parts.slice(0, i + 1).join('/')); }}, x)]),
      h('span', {class: 'muted'}, '   ' + GBf(d.total) + (d.running ? '   · a run is going: deleting waits until it ends' : '')));
    const big = Math.max(1, ...d.items.map(i => i.bytes || 0));
    put($('#ftable'), d.items.length ? h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, ''), h('th', {}, 'Name'), h('th', {}, ''), h('th', {class: 'n'}, 'Size'), h('th', {}, 'Changed'))),
      h('tbody', {}, d.items.map(i => {
        const cb = h('input', {type: 'checkbox', disabled: !!i.protected || d.running, title: i.protected ? 'Kept: ' + i.protected : '',
          onchange: () => { cb.checked ? F.chosen.add(i.rel) : F.chosen.delete(i.rel); updateSel(); }});
        return h('tr', {class: i.protected ? 'kept' : ''},
          h('td', {}, i.protected ? h('span', {class: 'lock', title: 'Kept: ' + i.protected}, 'KEEP') : cb),
          h('td', {}, i.kind === 'dir' ? h('a', {href: '#', onclick: e => { e.preventDefault(); loadDir(i.rel); }}, i.name + '/') : i.name,
            i.about ? h('div', {class: 'muted small'}, i.about) : i.protected ? h('div', {class: 'muted small'}, i.protected) : null),
          h('td', {class: 'sizebar'}, h('div', {class: 'bar'}, h('i', {style: `width:${100 * (i.bytes || 0) / big}%`}))),
          h('td', {class: 'n'}, GBf(i.bytes)),
          h('td', {class: 'muted'}, new Date(i.mtime * 1000).toLocaleDateString()));
      }))) : h('p', {class: 'muted', style: 'padding:16px'}, 'Empty folder.'));
  }
  function updateSel() {
    const n = F.chosen.size;
    const bytes = F.list ? F.list.items.filter(i => F.chosen.has(i.rel)).reduce((a, i) => a + (i.bytes || 0), 0) : 0;
    $('#fsel').textContent = n ? `${n} chosen · ${GBf(bytes)}` : 'Nothing chosen.';
    $('#fdelete').disabled = !n;
  }
  // one dialog for every delete: the plan, what is refused, the name when it is asked for
  function deleteDialog(title, body, wantName, go) {
    const d = $('#ddelete');
    $('#ddtitle').textContent = title;
    put($('#ddbody'), body);
    $('#ddname').hidden = !wantName;
    $('#ddwant').textContent = wantName || '';
    $('#ddinput').value = '';
    const gate = () => { $('#ddgo').disabled = !go || (wantName && $('#ddinput').value.trim() !== wantName); };
    $('#ddinput').oninput = gate;
    gate();
    $('#ddgo').textContent = 'Delete';
    $('#ddgo').onclick = async () => { $('#ddgo').disabled = true; await go($('#ddinput').value.trim()); };
    d.showModal();
  }
  const planList = items => h('ul', {class: 'plan'}, items.slice(0, 40).map(i => h('li', {}, h('span', {}, i.rel || i.path), h('span', {class: 'n'}, GBf(i.bytes)))),
    items.length > 40 ? h('li', {class: 'muted'}, `…and ${items.length - 40} more`) : null);
  $('#fdelete').addEventListener('click', async () => {
    const rels = [...F.chosen];
    let plan;
    try { plan = await api('plan_delete', {config: F.config, rels}); } catch (e) { toast(e.message, 'bad'); return; }
    const can = plan.items.length && !plan.refused.length;
    deleteDialog(`Delete ${plan.items.length} item(s), ${GBf(plan.bytes)}`, [
      h('p', {}, 'Deleted for good: there is no recycle bin inside Ubuntu.'),
      plan.items.length ? planList(plan.items) : null,
      plan.refused.length ? h('div', {class: 'note bad'}, 'Not allowed:', h('ul', {}, plan.refused.map(r => h('li', {}, r)))) : null,
      h('p', {class: 'muted small'}, 'Each deletion is written to the project\'s logs/deletions.jsonl.')],
      plan.needs_name ? plan.name : null,
      can ? async name => {
        try {
          const r = await api('delete_files', {config: F.config, rels, confirm_name: name});
          if (r.refused && r.refused.length) { toast(r.refused[0], 'bad'); return; }
          $('#ddelete').close();
          toast(`Freed ${GBf(r.freed)}` + (r.failed.length ? `; ${r.failed.length} could not be deleted` : '.'), r.failed.length ? 'warn' : '');
          loadDir(F.rel);
        } catch (e) { toast(e.message, 'bad'); }
      } : null);
  });
  async function spaceAction(action) {
    let plan;
    try { plan = await api('project_space', {config: F.config, action}); } catch (e) { toast(e.message, 'bad'); return; }
    const refused = plan.refused || [];
    const run = async (name, archive) => {
      try {
        const r = await api('project_space', {config: F.config, action, apply: true, confirm_name: name, archive});
        if (r.refused && r.refused.length) { toast(r.refused[0], 'bad'); return; }
        $('#ddelete').close();
        if (action === 'keep') { toast(`Freed ${GBf(r.freed)}.`); loadDir(''); }
        else { toast(`Deleted ${r.project}` + (r.archived ? `; archive in ${r.archive_dir}` : '') + '.'); location.hash = '#/files'; }
      } catch (e) { toast(e.message, 'bad'); }
    };
    if (action === 'keep') {
      deleteDialog(`Keep the results of ${F.name}`, [
        h('p', {}, `Deletes the processing files and keeps the deliverables, logs, releases and sign-off. Frees ${GBf(plan.bytes)}.`),
        plan.items.length ? planList(plan.items) : h('p', {class: 'muted'}, 'Nothing to free.'),
        (plan.notes || []).map(n => h('div', {class: 'note warn'}, n)),
        refused.length ? h('div', {class: 'note bad'}, refused.join(' ')) : null],
        null, plan.items.length && !refused.length ? name => run(name) : null);
      $('#ddgo').textContent = 'Keep results only';
    } else {
      deleteDialog(`Delete the project ${plan.project}`, [
        h('p', {}, `Deletes the whole folder, ${GBf(plan.bytes)}. First a slim archive (${plan.archive_files} files, ${GBf(plan.archive_bytes)}) is copied to ${plan.archive_dir}: the deliverables, the report, the logs and the config.`),
        (plan.skipped || []).length ? h('p', {class: 'muted small'}, `Left out of the archive (over 50 MB): ${plan.skipped.slice(0, 6).map(x => x[0]).join(', ')}${plan.skipped.length > 6 ? '…' : ''}`) : null,
        plan.why_name ? h('div', {class: 'note warn'}, 'This project ' + plan.why_name + '.') : null,
        (plan.notes || []).map(n => h('div', {class: 'note info'}, n)),
        refused.length ? h('div', {class: 'note bad'}, refused.join(' ')) : null],
        plan.project, !refused.length ? name => run(name, true) : null);
      $('#ddgo').textContent = 'Archive and delete';
    }
  }
  $('#fkeep').addEventListener('click', () => spaceAction('keep'));
  $('#fdelproj').addEventListener('click', () => spaceAction('delete'));
  $('#fopen').addEventListener('click', async () => { try { await api('open', {config: F.config, what: 'folder'}); } catch (e) { toast(e.message, 'bad'); } });
  $('#bfiles').addEventListener('click', () => { location.hash = '#/files/' + encodeURIComponent(P.name); });

  // ============================================================ export the map
  $('#bexport').addEventListener('click', () => {
    if (!P || !P.data) return;
    const on = Object.entries(layerState).filter(([, st]) => st.on && st.ready).map(([id]) => id);
    const labels = on.map(id => (layerState[id].meta || {}).label || id);
    $('#xtitle').value = $('#xtitle').value || P.name;
    $('#xbase').value = $$('#basemaps button.on')[0] ? $$('#basemaps button.on')[0].dataset.b : 'imagery';
    $('#xwhat').textContent = `The page shows what the live map shows now: this view, and ${on.length} layer(s): ${labels.join(', ')}.` +
      (P.data.results && P.data.results.final ? '' : ' The report is not FINAL, so the page is marked DRAFT.');
    $('#dexport').showModal();
  });
  $('#xgo').addEventListener('click', async () => {
    const b = pmap.getBounds();
    const on = Object.entries(layerState).filter(([, st]) => st.on && st.ready).map(([id]) => id);
    $('#xgo').disabled = true;
    const old = $('#xgo').textContent;
    $('#xgo').textContent = 'Drawing...';
    try {
      const r = await api('export', {config: P.config, page: $('#xpage').value, orientation: $('#xorient').value,
        bw: $('#xbw').value === '1', format: $('#xfmt').value, basemap: $('#xbase').value, logo: $('#xlogo').checked,
        title: $('#xtitle').value, subtitle: $('#xsub').value, notes: $('#xnotes').value, layers: on,
        bounds: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]});
      $('#dexport').close();
      toast(`Saved ${r.name} in the project's exports/maps folder; opening it.`);
      logActivity('Exported the map: ' + r.name);
    } catch (e) { toast(e.message, 'bad'); }
    $('#xgo').disabled = false; $('#xgo').textContent = old;
  });

  // ============================================================ the company
  let pendingLogo = null, removeLogo = false;
  async function loadCompany(body) {
    let c;
    try { c = await api('company', body || {}); } catch (e) { toast(e.message, 'bad'); return; }
    $('#cofirm').value = c.firm || ''; $('#coprep').value = c.preparer || ''; $('#cocontact').value = c.contact || '';
    $('#cologo').hidden = !c.logo; $('#cologonone').hidden = !!c.logo;
    if (c.logo) $('#cologo').src = c.logo;
    pendingLogo = null; removeLogo = false;
    return c;
  }
  $('#cofile').addEventListener('change', () => {
    const f = $('#cofile').files[0];
    if (!f) return;
    if (f.size > 5e6) { toast('That logo is over 5 MB.', 'bad'); return; }
    const rd = new FileReader();
    rd.onload = () => { pendingLogo = rd.result; removeLogo = false; $('#cologo').src = rd.result; $('#cologo').hidden = false; $('#cologonone').hidden = true; };
    rd.readAsDataURL(f);
  });
  $('#coremove').addEventListener('click', () => { removeLogo = true; pendingLogo = null; $('#cologo').hidden = true; $('#cologonone').hidden = false; });
  $('#cosave').addEventListener('click', async () => {
    const body = {firm: $('#cofirm').value, preparer: $('#coprep').value, contact: $('#cocontact').value};
    if (pendingLogo) body.logo_data = pendingLogo;
    if (removeLogo) body.remove_logo = true;
    if (await loadCompany(body)) toast('Saved. The next export carries it.');
  });

  // ============================================================ computer
  async function showComputer() {
    loadCompany();
    every(4000, async () => {
      try {
        const r = await api('resources', {});
        const d = await api('home');
        const live = d.projects.filter(p => p.state === 'running');
        put($('#cres'), h('h2', {}, 'Now'), ...resourceBlocks(r),
          h('h2', {}, 'Runs going'),
          live.length ? h('div', {class: 'areas'}, live.map(p => h('a', {class: 'area', href: '#/p/' + encodeURIComponent(p.name), style: 'text-decoration:none;color:inherit'},
            h('span', {class: 'n'}, p.name), h('span', {class: 'r'}, p.live ? (p.live.remaining || '') : ''),
            h('span', {class: 'rd muted'}, p.live ? p.live.label : 'running')))) : h('p', {class: 'muted'}, 'None.'));
      } catch (e) { toast(e.message, 'bad'); }
    });
  }
  $('#loadstorage').addEventListener('click', async () => {
    put($('#storage'), h('p', {class: 'muted'}, 'Measuring every project folder…'));
    try {
      const d = await api('storage');
      put($('#storage'), h('pre', {style: 'padding:0 16px 16px;white-space:pre-wrap;font:12px/1.5 var(--mono)'}, d.overview));
    } catch (e) { put($('#storage'), h('div', {class: 'note bad', style: 'margin:12px 16px'}, e.message)); }
  });
  $('#quit').addEventListener('click', async () => {
    if (!await confirmBox('Quit the app?', 'This closes the app on this computer. Runs keep going; open the app again from the InSAR Pipeline icon to watch them.', 'Quit')) return;
    try { await api('quit'); } catch (e) { /* it is going */ }
    put(document.body, h('div', {class: 'empty', style: 'margin:60px auto;max-width:520px'}, 'The app has stopped. Runs carry on. Close this window; open the InSAR Pipeline icon to come back.'));
  });

  // ============================================================ the pointer's ring, the passing satellite
  // A click on a banner or a button sends one holographic ripple out from where
  // it landed. The ripple stays at that spot while it spreads (it does not follow
  // the pointer) and is removed when it has faded. Sized to reach the farthest corner.
  const stillR = matchMedia('(prefers-reduced-motion: reduce)');
  function ripple(el, x, y) {
    const b = el.getBoundingClientRect(), px = x - b.left, py = y - b.top;
    const R = Math.ceil(Math.hypot(Math.max(px, b.width - px), Math.max(py, b.height - py))) + 24;
    const r = document.createElement('i');
    r.className = 'ripple';
    r.setAttribute('aria-hidden', 'true');
    Object.assign(r.style, {left: (px - R) + 'px', top: (py - R) + 'px', width: 2 * R + 'px', height: 2 * R + 'px'});
    el.prepend(r);                                     // z-index -1 in an isolated box: under the text (app.css)
    const gone = () => r.remove();
    r.addEventListener('animationend', gone, {once: true});
    setTimeout(gone, 1600);                            // in case the animation never runs
  }
  addEventListener('pointerdown', e => {
    if (stillR.matches || e.button > 0) return;
    const el = e.target.closest && e.target.closest('.pagehead, .nowcard, .dlgbody > header, button:not(.ghost):not(:disabled)');
    if (el) ripple(el, e.clientX, e.clientY);
  }, {passive: true});

  const SAT = ['####..###..####', '#..#..#.#..#..#', '###############', '#..#..#.#..#..#', '####..###..####',
               '......###......', '.......#.......', '.....#####.....'];
  const BEEPS = ['beep beep beep', 'beep. beep. beep.', 'BEEP BEEP BEEP', 'bip bip bip', 'beep beep... boop?', 'beep (12 days later) beep',
                 'beep beep: still coherent', 'beep beep beep beep', '...beep', 'acquiring. beep.', 'beep beep, phase wrapped', 'b-e-e-p'];
  const still2 = matchMedia('(prefers-reduced-motion: reduce)');

  // ORBIT-BEGIN (tests/test_app_orbit.py runs this block in node)
  // Each pass is a real orbit. Random elements are drawn (altitude, inclination,
  // eccentricity, argument of perigee), the satellite is put on Kepler's
  // two-body orbit, and its ground track is followed over a random patch of
  // the globe that the window stands for, with the Earth turning under it.
  // Speeds come from vis-viva, the curve from the track itself, so low orbits
  // dash across, navigation orbits drift, polar ones cross north-south and
  // retrograde ones run west. One time compression for every pass.
  const ORBIT = (() => {
    const GM = 398600.4418, RE = 6378.137, WE = 7.2921159e-5, D = Math.PI / 180, WARP = 40;
    const KINDS = [   // [weight, name, perigee altitude km (lo, hi), inclination deg (lo, hi), eccentricity (lo, hi)]
      [35, 'radar, sun-synchronous', [650, 770], [97.6, 98.8], [0, 0.002]],
      [15, 'station orbit', [380, 440], [51.6, 51.7], [0, 0.001]],
      [30, 'low orbit', [450, 1350], [0, 110], [0, 0.02]],
      [12, 'navigation orbit', [19000, 23000], [53, 65], [0, 0.01]],
      [8, 'eccentric orbit', [300, 1200], [10, 80], [0.05, 0.2]],
    ];
    const pick = (rnd, [lo, hi]) => lo + (hi - lo) * rnd();
    function kepler(M, e) {                       // eccentric anomaly E from mean anomaly M (Newton)
      let E = e < 0.8 ? M : Math.PI;
      for (let k = 0; k < 20; k++) {
        const d = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
        E -= d;
        if (Math.abs(d) < 1e-12) break;
      }
      return E;
    }
    function draw(rnd = Math.random, aspect = 16 / 9) {
      let w = rnd() * KINDS.reduce((t, k) => t + k[0], 0), K = KINDS[0];
      for (const k of KINDS) { if ((w -= k[0]) <= 0) { K = k; break; } }
      const hp = pick(rnd, K[2]), inc = pick(rnd, K[3]) * D, e = pick(rnd, K[4]);
      const a = (RE + hp) / (1 - e), n = Math.sqrt(GM / a ** 3), argp = rnd() * 2 * Math.PI;
      // the window: a patch of the globe span degrees wide, centred where this orbit can reach
      const span = 45 + rnd() * 40, maxLat = Math.min(inc, Math.PI - inc) / D;
      const latC = (rnd() * 2 - 1) * Math.min(maxLat * 0.85, 65);
      // put the satellite over the window's centre at t = 0, going up or coming down
      const s = Math.max(-1, Math.min(1, Math.sin(latC * D) / Math.max(Math.sin(inc), 1e-9)));
      const u0 = rnd() < 0.5 ? Math.asin(s) : Math.PI - Math.asin(s);
      const nu0 = u0 - argp;
      const E0 = 2 * Math.atan2(Math.sqrt(1 - e) * Math.sin(nu0 / 2), Math.sqrt(1 + e) * Math.cos(nu0 / 2));
      const M0 = E0 - e * Math.sin(E0);
      const node = -Math.atan2(Math.cos(inc) * Math.sin(u0), Math.cos(u0));    // longitude of the node, window at 0
      function at(t) {
        const E = kepler(M0 + n * t, e);
        const nu = 2 * Math.atan2(Math.sqrt(1 + e) * Math.sin(E / 2), Math.sqrt(1 - e) * Math.cos(E / 2));
        const r = a * (1 - e * Math.cos(E)), u = argp + nu;
        const lat = Math.asin(Math.sin(inc) * Math.sin(u));
        let lon = node + Math.atan2(Math.cos(inc) * Math.sin(u), Math.cos(u)) - WE * t;
        lon = Math.atan2(Math.sin(lon), Math.cos(lon));
        return {x: lon / D * Math.cos(latC * D) / span, y: -(lat / D - latC) / (span / aspect),   // -0.5..0.5 is on screen
                alt: r - RE, v: Math.sqrt(GM * (2 / r - 1 / a)), lat: lat / D, lon: lon / D};
      }
      // follow the track out of the window both ways (a little past the edges)
      const T = 2 * Math.PI / n, dt = T / 4000, out = p => Math.abs(p.x) > 0.62 || Math.abs(p.y) > 0.62;
      let t0 = 0, t1 = 0;
      while (t0 > -T / 2 && !out(at(t0))) t0 -= dt;
      while (t1 < T / 2 && !out(at(t1))) t1 += dt;
      let dur = (t1 - t0) / WARP;
      const cap = 75;                               // seconds on screen at most: a slow orbit shows part of its pass
      if (dur > cap) { const mid = (t0 + t1) / 2; t0 = mid - cap * WARP / 2; t1 = mid + cap * WARP / 2; dur = cap; }
      return {kind: K[1], perigee_km: hp, inc_deg: inc / D, e, period_min: T / 60, t0, t1, seconds: dur, at, span, latC};
    }
    return {draw, kepler, GM, RE, WARP};
  })();
  // ORBIT-END

  // ?satellite flies one at once; ?satellite=<n> flies the same orbit every time (n seeds it)
  const satArg = (location.search.match(/[?&]satellite(?:=(\d+))?/) || []);
  let seed = satArg[1] ? Number(satArg[1]) : 0;
  const seeded = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  function pass() {
    if (!still2.matches && !document.hidden) fly(ORBIT.draw(seed ? seeded : Math.random, innerWidth / Math.max(1, innerHeight)));
    setTimeout(pass, 120000 + Math.random() * 120000);    // every two to four minutes
  }
  // What the satellite reports: the project on screen, else the run going now,
  // else the most recently updated project; its live stage or its headline
  // figures (the same rows as the project cards: api home's areas, at most two).
  function downlink(list) {
    const here = route.view === 'project' ? list.find(p => p.name === route.name) : null;
    const live = list.filter(p => !p.gone && p.state === 'running').sort((a, b) => b.updated_t - a.updated_t);
    const p = here || live[0] || list.filter(q => !q.gone).sort((a, b) => b.updated_t - a.updated_t)[0];
    if (!p) return null;
    const lines = [];
    if (p.live) lines.push(`${p.live.label}${p.live.total ? ` ${p.live.done}/${p.live.total}` : ''}${p.live.remaining ? ` · ${p.live.remaining} left` : ''}`);
    for (const a of (p.areas || []).slice(0, 2)) lines.push(`${a.name}: ${a.rate}${a.reading ? ' · ' + a.reading : ''}`);
    if (p.send_rule) lines.push(`Verdict: ${p.send_rule}${p.final ? ' (final)' : p.final === false ? ' (draft)' : ''}`);
    if (!lines.length) lines.push(p.error ? 'Needs attention' : 'No results yet');
    return {name: p.name, state: STATE_WORD[p.state] || p.state, lines};
  }
  async function fly(o) {
    const sky = $('#sky');
    const mid = o.at((o.t0 + o.t1) / 2);
    let list = projects;
    if (!list.length || route.view !== 'projects') { try { list = (await api('home')).projects || []; } catch (e) { /* offline: beeps only */ } }
    const dl = downlink(list || []);
    const words = BEEPS[Math.floor(Math.random() * BEEPS.length)];
    const facts = `${o.kind} · ${Math.round(mid.alt).toLocaleString()} km · i ${o.inc_deg.toFixed(1)}° · ${mid.v.toFixed(1)} km/s`;
    const body = h('div', {class: 'body'}, sprite(SAT));
    const say = dl
      ? h('div', {class: 'say'}, h('b', {}, h('i', {class: 'led'}), `${dl.name} · ${dl.state}`),
          dl.lines.map(t => h('span', {class: 'ln'}, t)), h('small', {}, `${words} · ${facts}`))
      : h('div', {class: 'say'}, h('b', {}, h('i', {class: 'led'}), words), h('small', {}, facts));
    const sat = h('div', {class: 'sat'}, say, body, h('div', {class: 'beam'}));
    sky.append(sat);
    const scale = Math.max(0.5, Math.min(1.3, Math.pow(700 / Math.max(mid.alt, 200), 0.3)));   // higher looks smaller
    body.style.width = Math.round(90 * scale) + 'px';
    // the orbit point is the sprite's centre, wherever the label above it puts the sprite
    const cx = () => sat.offsetWidth / 2, cy = () => body.offsetTop + body.offsetHeight / 2;
    const W = innerWidth, H = innerHeight, start = performance.now(), ms = o.seconds * 1000;
    let last = 0;
    const step = now => {
      const p = (now - start) / ms;
      if (p >= 1 || !sat.isConnected) { sat.remove(); return; }
      if (now - last >= 90) {                                 // pixel time: the position moves in steps
        last = now;
        const t = o.t0 + p * (o.t1 - o.t0), q = o.at(t), q2 = o.at(t + (o.t1 - o.t0) / 200);
        const x = Math.round((0.5 + q.x) * W / 3) * 3, y = Math.round((0.5 + q.y) * H / 3) * 3;
        const ang = Math.round(Math.atan2((q2.y - q.y) * H, (q2.x - q.x) * W) / (Math.PI / 12)) * 15;   // faces its way, 15° steps
        sat.style.transform = `translate(${x - Math.round(cx())}px, ${y - Math.round(cy())}px)`;
        body.style.transform = `rotate(${ang}deg)`;
        sat.style.opacity = String(Math.min(1, p * 12, (1 - p) * 12));   // a slow pass fades in and out mid-screen
      }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  setTimeout(pass, satArg.length ? 300 : 25000 + Math.random() * 20000);

  go();
})();
