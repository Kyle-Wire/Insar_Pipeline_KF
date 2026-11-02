"""The app's passing satellite flies a real orbit (app_assets/app.js, the
ORBIT block, run here in node): Kepler's equation, vis-viva speeds, a ground
track over the window's patch of the globe with the Earth turning under it."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parents[1] / "app_assets" / "app.js"
NODE = shutil.which("node")

PROBE = r"""
let seed = 12345;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const out = {kinds: {}, passes: []};
for (let i = 0; i < 400; i++) {
  const o = ORBIT.draw(rnd, 16 / 9);
  const c = o.at(0), a = o.at(o.t0), b = o.at(o.t1), m = o.at((o.t0 + o.t1) / 2);
  const q = o.at(o.t0 + (o.t1 - o.t0) * 0.45), q2 = o.at(o.t0 + (o.t1 - o.t0) * 0.55);
  out.kinds[o.kind] = (out.kinds[o.kind] || 0) + 1;
  out.passes.push({kind: o.kind, inc: o.inc_deg, e: o.e, per: o.period_min, sec: o.seconds, alt: m.alt, v: m.v,
                   c: [c.x, c.y], a: [a.x, a.y], b: [b.x, b.y], dlon: q2.lon - q.lon, dlat: q2.lat - q.lat});
}
const E = ORBIT.kepler(2.0, 0.3);
out.kepler = E - 0.3 * Math.sin(E);
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def flights():
    if NODE is None:
        pytest.skip("node is not installed")
    src = JS.read_text()
    block = src[src.index("// ORBIT-BEGIN"):src.index("// ORBIT-END")]
    r = subprocess.run([NODE, "-e", block + PROBE], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_kepler_and_vis_viva(flights):
    assert flights["kepler"] == pytest.approx(2.0, abs=1e-10)          # E - e sin E = M
    radar = [p for p in flights["passes"] if p["kind"].startswith("radar")]
    assert radar
    for p in radar:                                                     # ~700 km: ~7.5 km/s, ~99 min
        assert 7.4 < p["v"] < 7.6 and 97 < p["per"] < 101 and 96 < p["inc"] < 99


def test_every_pass_crosses_the_window_through_its_middle(flights):
    for p in flights["passes"]:
        assert abs(p["c"][0]) < 1e-6 and abs(p["c"][1]) < 1e-6         # over the centre at t = 0
        if p["sec"] < 75:                                               # a whole pass: it enters and leaves
            assert max(abs(v) for v in p["a"]) > 0.6 and max(abs(v) for v in p["b"]) > 0.6


def test_orbits_differ_the_way_physics_says(flights):
    P = flights["passes"]
    assert len(flights["kinds"]) == 5
    leo = [p["sec"] for p in P if p["kind"] in ("radar, sun-synchronous", "station orbit", "low orbit")]
    meo = [p["sec"] for p in P if p["kind"] == "navigation orbit"]
    assert sorted(leo)[len(leo) // 2] < 40 < sorted(meo)[len(meo) // 2]   # low orbits dash, navigation ones drift
    for p in P:
        if p["kind"].startswith("radar"):                                # near-polar: mostly north-south over the ground
            assert abs(p["dlat"]) > abs(p["dlon"]) * 0.5
        if p["kind"] == "low orbit" and p["inc"] < 30 and p["e"] < 0.01:  # low inclination: eastward
            assert p["dlon"] > 0
    ecc = [p for p in P if p["kind"] == "eccentric orbit"]
    assert ecc and all(p["alt"] > 200 for p in P)                         # never underground
