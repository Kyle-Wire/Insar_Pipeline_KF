"""The builder's step 0 (A5 of review_2026-11-07/PLAN_oil_gas_slopes.md): the
study list, the request the page sends (the BODY block of project_builder.html,
run here in node), and the create call reaching areas.write_project with the
study and its layers. The builder stays map-only: no processing key in any
study's request (Gate A5(a)), and the default study's request and call are
today's."""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from insar_pipeline import areas, domains, map_builder as mb, runs, settings

PKG = Path(__file__).resolve().parents[1]
HTML = (PKG / "project_builder.html").read_text()
# node on the PATH, else the one the insar environment installs beside python
NODE = shutil.which("node") or next((str(p) for p in [Path(sys.executable).parent / "node"] if p.exists()), None)

# today's request (project_builder.html before step 0, 9 Nov 2026): payload()
# and the body literal of create(), copied as they were
OLD_JS = r"""
function oldPayload(l) {
  const base = {role: l.role, kind: l.kind, name: l.name, report: l.report, source: l.source || '',
                ...(l.reopened ? {reopened: true} : {}),
                ...(l.kind === 'dated' ? {year_field: l.year_field || '', name_field: l.name_field || ''} : {})};
  return l.upload_id ? {...base, upload_id: l.upload_id} : {...base, geojson: l.geojson};
}
function oldBody(f, layers, overwrite) {
  return {name: f.name, route: f.route, start: f.start, end: f.end, pad_km: f.pad_km,
          track: f.track, direction: f.direction, disp_frame: f.disp_frame, reference: f.reference,
          layers: layers.map(oldPayload), overwrite};
}
"""
OLD_KEYS = {"name", "route", "start", "end", "pad_km", "track", "direction", "disp_frame", "reference", "layers",
            "overwrite"}
LAYER_KEYS = {"role", "kind", "name", "report", "source", "reopened", "year_field", "name_field", "upload_id",
              "geojson"}

PROBE = r"""
const gj = {type: 'FeatureCollection', features: [{type: 'Feature', properties: {},
            geometry: {type: 'Polygon', coordinates: [[[-87.5, 38.3], [-87.4, 38.3], [-87.4, 38.4], [-87.5, 38.3]]]}}]};
const layers = [
  {role: 'area', kind: null, name: 'Zone', geojson: gj, report: false, source: ''},
  {role: 'mine', kind: 'underground', name: 'CMIS', upload_id: 'ab'.repeat(16), report: true, source: 'State maps, 2023'},
  {role: 'mine', kind: 'dated', name: 'Panels', geojson: gj, report: false, source: '', year_field: 'YR', name_field: ''},
  {role: 'mine', kind: null, name: 'mapped mines', geojson: gj, report: false, source: 'x', reopened: true},
  {role: 'report', kind: null, name: 'North field', geojson: gj, report: false, source: ''}];
const form = {name: 'Test Mine', route: 'miaplpy', start: '2016-01-01', end: '2026-06-30', pad_km: 5, track: 19,
              direction: 'ASCENDING', disp_frame: '', reference: '38.33,-87.47'};
const out = {default_new: [], default_old: [], studies: {}};
for (const ov of [false, true]) {
  out.default_new.push(JSON.stringify(createBody(form, layers, DEFAULT_STUDY, ov)));
  out.default_new.push(JSON.stringify(createBody(form, layers, undefined, ov)));
  out.default_old.push(JSON.stringify(oldBody(form, layers, ov)));
  out.default_old.push(JSON.stringify(oldBody(form, layers, ov)));
}
const pt = {type: 'FeatureCollection', features: [{type: 'Feature', properties: {status: 'PA'},
            geometry: {type: 'Point', coordinates: [-87.45, 38.35]}}]};
for (const k of STUDY_KEYS) {
  const sl = [layers[0], layers[4], {role: 'study', kind: 'wells', name: 'w', geojson: pt},
              {role: 'study', kind: 'pads', name: 'p', upload_id: 'cd'.repeat(16)}];
  out.studies[k] = createBody(form, sl, k, false);
}
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def bodies():
    if NODE is None:
        pytest.skip("node is not installed")
    block = HTML[HTML.index("// BODY-BEGIN"):HTML.index("// BODY-END")]
    head = "const DEFAULT_STUDY = 'ground_subsidence';\nconst STUDY_KEYS = %s;\n" % json.dumps(list(domains.STUDY_KEYS))
    r = subprocess.run([NODE, "-e", head + block + OLD_JS + PROBE], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_the_page_uses_the_tested_block():
    # create() builds its request with createBody (the block tested here) and
    # names the same default study as domains.py
    assert "const body = createBody(" in HTML and "layers: state.layers.map(payload)" not in HTML
    assert f"const DEFAULT_STUDY = '{domains.DEFAULT_STUDY}'" in HTML


def test_default_study_request_is_byte_for_byte_todays(bodies):
    """Gate: the default study posts exactly what the builder posted before step 0."""
    assert bodies["default_new"] == bodies["default_old"]
    assert set(json.loads(bodies["default_new"][0])) == OLD_KEYS


def test_no_processing_key_in_any_studys_request(bodies):
    """Gate A5(a): for every study, the request is today's keys plus the study
    name, and each layer carries only its geometry and naming."""
    processing = set(domains.PRESET_KEYS) | set(domains.NEVER_PRESET) | {
        "event_products", "feature_layers", "event_polygons", "event_lines", "fast_zone_geojson",
        "reference_stable_geojson", "site_profile", "mintpy_deramp", "reference_mode", "study_layers"}
    for k, body in bodies["studies"].items():
        extra = {"study"} if k != domains.DEFAULT_STUDY else set()
        assert set(body) == OLD_KEYS | extra, k
        if extra:
            assert body["study"] == k
        for L in body["layers"]:
            assert set(L) <= LAYER_KEYS, (k, L)
        study_rows = [L for L in body["layers"] if L["role"] == "study"]
        assert [set(L) for L in study_rows] == [{"role", "kind", "name", "geojson"}, {"role", "kind", "name", "upload_id"}]
        assert not (set(body) | {x for L in body["layers"] for x in L}) & processing


def test_step_zero_lists_default_then_oil_gas_then_the_rest_greyed():
    d = mb.api_studies({})
    st = d["studies"]
    assert d["default"] == domains.DEFAULT_STUDY
    assert st[0]["key"] == domains.DEFAULT_STUDY and st[0]["selectable"] and st[0]["badge"] is None
    assert st[1]["key"] == "oil_gas" and st[1]["selectable"] and st[1]["badge"] == "Screening"
    assert {x["key"] for x in st} == set(domains.STUDY_KEYS)
    for x in st[2:]:
        assert not x["selectable"] and x["reason"], x["key"]
    slopes = next(x for x in st if x["key"] == "slopes")
    assert slopes["reason"] == domains.study("slopes").builder_reason
    og = st[1]
    assert [k["key"] for k in og["layer_kinds"]] == [L.key for L in domains.OIL_GAS.layer_kinds]
    assert og["layer_kinds"][0]["geometry"] == "point" and og["layer_kinds"][0]["attributes"] == ["status", "plug_date", "type"]
    # the two step-6 cards: every output with its tier, every cannot and not-yet line
    assert len(og["will_give"]) == len(domains.OIL_GAS.outputs) and all(":" in t for t in og["will_give"])
    assert len(og["cannot"]) == len(domains.OIL_GAS.cannot) and len(og["not_yet"]) == len(domains.OIL_GAS.not_yet)
    assert "will_give" not in st[0]                                     # the default study shows no cards


# ---------------------------------------------------------------- the create call
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "projects_root", lambda: tmp_path / "projects")
    monkeypatch.setattr(settings, "configs_dir", lambda: tmp_path / "configs")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(mb, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(areas, "subswaths", lambda *a, **k: [2, 3])
    monkeypatch.setattr(areas, "burst_coverage", lambda area, track, direction, targets=None, **k: {
        "swaths": [2, 3], "bursts": 4, "covered": 1.0, "targets": {n: 1.0 for n in (targets or {})}})
    calls = []
    real = areas.write_project

    def spy(*a, **k):
        calls.append(k)
        return real(*a, **k)
    monkeypatch.setattr(areas, "write_project", spy)
    return tmp_path, calls


AREA = {"role": "area", "name": "Zone", "geojson": areas.box(38.29, -87.52, 38.37, -87.42, "Zone").geojson()}
BASE = {"route": "miaplpy", "start": "2016-01-01", "end": "2026-06-30", "track": 19, "direction": "ASCENDING",
        "reference": "", "pad_km": 5}
# the keyword arguments today's builder gives write_project
TODAY_KWARGS = {"track", "direction", "disp_frame", "swaths", "report_areas", "mine_sources", "file_stems", "overwrite",
                "deforming", "underground", "surface", "dated", "pad_km", "reference_lalo"}


def _wells_upload():
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"api": f"W{i}", "status": "PA" if i % 2 else "Active",
                                           "plug_date": "2019-05-01" if i % 2 else "", "type": "oil"},
         "geometry": {"type": "Point", "coordinates": [-87.47 + 0.005 * i, 38.33]}} for i in range(4)]
        + [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
            [-87.5, 38.3], [-87.49, 38.3], [-87.49, 38.31], [-87.5, 38.3]]]}}]}
    data = base64.b64encode(json.dumps(fc).encode()).decode()
    return mb.api_outline({"files": [{"name": "wells.geojson", "data": data}], "study": "oil_gas",
                           "study_kind": "wells"})


def test_default_study_create_calls_write_project_as_today(sandbox):
    _, calls = sandbox
    r = mb.api_create({**BASE, "name": "Test Mine", "layers": [AREA]})
    assert set(calls[0]) == TODAY_KWARGS                                 # no study, no study_layers
    assert "study" not in r and "study_layers" not in r
    saved = json.loads((Path(r["project_dir"]) / "context" / "builder_layers.json").read_text())
    assert set(saved) == {"name", "reference", "pad_km", "layers"}


def test_a_study_layer_upload_keeps_its_columns(sandbox):
    up = _wells_upload()
    assert up["features"] == 4 and up["geometry"] == "point"            # the polygon is not a well
    assert up["found"] == ["status", "plug_date", "type"] and up["missing"] == []
    assert "api" in up["columns"]
    with pytest.raises(ValueError, match="needs lines"):
        data = base64.b64encode(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [-87.4, 38.3]}}]}).encode())
        mb.api_outline({"files": [{"name": "f.geojson", "data": data.decode()}], "study": "oil_gas",
                        "study_kind": "faults"})


def test_oil_gas_create_reaches_write_project_with_study_and_layers(sandbox):
    tmp, calls = sandbox
    up = _wells_upload()
    pads = areas.box(38.32, -87.47, 38.34, -87.45).geojson()
    pads["features"][0]["properties"] = {"start": "2018-01-01", "end": "2019-01-01"}
    fault = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
             "geometry": {"type": "LineString", "coordinates": [[-87.5, 38.3], [-87.43, 38.36]]}}]}
    body = {**BASE, "name": "Field One", "study": "oil_gas",
            "layers": [AREA, {"role": "study", "kind": "wells", "name": "wells", "upload_id": up["upload_id"]},
                       {"role": "study", "kind": "pads", "name": "Pads 1 (drawn)", "geojson": pads},
                       {"role": "study", "kind": "faults", "name": "Faults 1 (drawn)", "geojson": fault}]}
    r = mb.api_create(body)
    k = calls[-1]
    assert set(k) == TODAY_KWARGS | {"study", "study_layers"} and k["study"] == "oil_gas"
    assert set(k["study_layers"]) == {"wells", "pads", "faults"}
    wells = k["study_layers"]["wells"]["features"]
    assert len(wells) == 4 and wells[1]["properties"]["status"] == "PA"   # read again in full, with its columns
    from insar_pipeline.config import load_config
    cfg = load_config(r["config"])
    assert cfg.study == "oil_gas" and cfg.event_polygons and cfg.event_products
    assert sorted(x.split(":", 1)[0] for x in cfg.feature_layers) == ["faults", "wells"]
    assert r["study"] == "Oil and gas fields"
    assert {x["kind"]: x["features"] for x in r["study_layers"]} == {"wells": 4, "pads": 1, "faults": 1}
    # reopened with the study and its layers (their columns kept)
    again = mb.api_project({"config": r["config"]})
    assert again["study"] == "oil_gas"
    sl = [L for L in again["layers"] if L["role"] == "study"]
    assert [L["kind"] for L in sl] == ["wells", "pads", "faults"]
    assert sl[0]["geojson"]["features"][1]["properties"]["plug_date"] == "2019-05-01"


def test_create_refusals_reach_the_page(sandbox):
    pt = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
          "geometry": {"type": "Point", "coordinates": [-80.0, 30.0]}}]}
    far = {"role": "study", "kind": "wells", "name": "w", "geojson": pt}
    with pytest.raises(ValueError, match="no feature lies inside the processing box"):   # write_project's own
        mb.api_create({**BASE, "name": "Field Two", "study": "oil_gas", "layers": [AREA, far]})
    with pytest.raises(ValueError, match="does not offer"):
        mb.api_create({**BASE, "name": "Field Two", "study": "basin_subsidence", "layers": [AREA]})
    with pytest.raises(ValueError, match="does not offer"):
        mb.api_create({**BASE, "name": "Field Two", "study": "slopes", "layers": [AREA]})
    mine = {"role": "mine", "kind": "surface", "name": "Pit", "geojson": areas.box(38.33, -87.46, 38.34, -87.45).geojson()}
    with pytest.raises(ValueError, match="mine layers belong"):
        mb.api_create({**BASE, "name": "Field Two", "study": "oil_gas", "layers": [AREA, mine]})
    with pytest.raises(ValueError, match="takes mine layers"):
        mb.api_create({**BASE, "name": "Field Two", "layers": [AREA, {**far, "kind": "wells"}]})
    with pytest.raises(ValueError, match="no layer 'casing'"):
        mb.api_create({**BASE, "name": "Field Two", "study": "oil_gas", "layers": [AREA, {**far, "kind": "casing"}]})
