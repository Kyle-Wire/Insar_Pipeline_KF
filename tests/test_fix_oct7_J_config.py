"""7 Oct 2026 fixes, group J (config.py, areas.py, map_builder.py,
project_builder.html): reopening a project in the builder keeps its mine
types, sources and record window (#55, #56); a relative report-area polygon
is refused at load (#58); a refused update puts context/ back (#59)."""
import json
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from insar_pipeline import areas, map_builder as mb, runs, settings
from insar_pipeline.config import ConfigError, load_config

PKG = Path(__file__).resolve().parents[1]
HTML = (PKG / "project_builder.html").read_text()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "projects_root", lambda: tmp_path / "projects")
    monkeypatch.setattr(settings, "configs_dir", lambda: tmp_path / "configs")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(mb, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(areas, "subswaths", lambda *a, **k: [2])
    # the weather default depends on this machine (is raider-env installed?); these tests are about
    # the builder, so the projects they make keep their own dates (test_fix_oct24_hrrr_default.py)
    monkeypatch.setattr(areas, "weather_default", lambda route, bbox, start: ({"mintpy_tropo_method": "pyaps"}, start, ""))
    monkeypatch.setattr(areas, "burst_coverage", lambda area, track, direction, targets=None, **k: {
        "swaths": [2], "bursts": 4, "covered": 1.0, "targets": {n: 1.0 for n in (targets or {})}})
    return tmp_path


ZONE = areas.box(38.29, -87.52, 38.37, -87.42, "Zone")
MINES = areas.box(38.33, -87.46, 38.34, -87.45, "mines")


def _legacy_project(sandbox, keep=()) -> Path:
    """A project like Site B: written before mine types and the builder -
    one untyped layer (deramp_exclude_geojson), no mine_layer_sources, no
    builder_layers.json. `keep`: typed fields to leave in the config."""
    out = areas.write_project("Legacy Site", ZONE, "miaplpy", "2016-01-01", "2026-06-30", track=19,
                              direction="ascending", swaths=[2],
                              report_areas=[("Full Satellite Imaging", "full"), ("Zone", ZONE)],
                              underground=MINES, deforming=MINES,
                              mine_sources={"underground": "State mine maps, 2016", "other": "Site survey, 2024"},
                              projects_root=sandbox / "projects", configs_dir=sandbox / "configs")
    doc = yaml.safe_load(out.read_text())
    for k in ("underground_mines_geojson", "other_movement_geojson", "mine_layer_sources"):
        if k not in keep:
            doc.pop(k, None)
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    load_config(str(out))
    return out


# ---------------------------------------------------------------- #55
def test_untyped_legacy_mine_layer_reopens_without_a_type_or_a_drawn_today_source(sandbox):
    out = _legacy_project(sandbox)
    d = mb.api_project({"config": str(out)})
    mine = [L for L in d["layers"] if L["role"] == "mine"]
    assert len(mine) == 1
    # not 'other known movement': the report would say the mines are "ground
    # already known to move"
    assert mine[0]["kind"] is None and mine[0]["reopened"]
    assert mine[0]["source"] == "source and date not recorded"
    body = {"name": "Legacy Site", "route": "miaplpy", "track": 19, "direction": "ASCENDING",
            "start": d["start"], "end": d["end"], "overwrite": True, "layers": d["layers"]}
    # Update waits for a type (the page disables it; the server refuses too)
    with pytest.raises(ValueError, match="type of the mine layer"):
        mb.api_create(body)
    mine[0]["kind"] = "underground"
    r = mb.api_create(body)
    cfg = load_config(r["config"])
    assert cfg.underground_mines_geojson and not cfg.other_movement_geojson
    assert list(cfg.mine_layer_sources) == ["Underground mine workings: source and date not recorded"]
    assert not any("drawn on" in s for s in cfg.mine_layer_sources)


def test_typed_layers_reopen_with_the_sources_the_config_records(sandbox):
    # a dashboard- or hand-written config (no builder_layers.json) with an
    # underground layer AND other known movement: both come back, each with
    # its own source (before, the other-movement layer was dropped and both
    # were credited as drawn on the builder map that day)
    out = _legacy_project(sandbox, keep=("underground_mines_geojson", "other_movement_geojson", "mine_layer_sources"))
    d = mb.api_project({"config": str(out)})
    mine = {L["kind"]: L for L in d["layers"] if L["role"] == "mine"}
    assert set(mine) == {"underground", "other"}
    assert mine["underground"]["source"] == "State mine maps, 2016"
    assert mine["other"]["source"] == "Site survey, 2024"
    r = mb.api_create({"name": "Legacy Site", "route": "miaplpy", "track": 19, "direction": "ASCENDING",
                       "start": d["start"], "end": d["end"], "overwrite": True, "layers": d["layers"]})
    assert r["mine_layer_sources"] == ["Underground mine workings: State mine maps, 2016",
                                       "Other known ground movement: Site survey, 2024"]


def test_layer_source_of_a_reopened_layer_is_never_drawn_today():
    assert mb._layer_source({"role": "mine", "source": "", "reopened": True}) == "source and date not recorded"
    assert mb._layer_source({"role": "mine", "source": "DNR, 2020", "reopened": True}) == "DNR, 2020"
    assert mb._layer_source({"role": "mine", "source": ""}).startswith("drawn on the project builder map")


def test_builder_page_keeps_a_reopened_untyped_layer_untyped():
    # the page used to turn kind null into 'other' and let Update through
    assert "if (l.role === 'mine' && !l.kind) return" in HTML                       # layerProblem
    assert "kind = role === 'mine' ? (kind || (reopened ? null : 'other')) : null;" in HTML
    assert "reopened: true}));" in HTML                                             # openSel
    assert "...(l.reopened ? {reopened: true} : {})" in HTML                        # payload
    assert "Choose the type…" in HTML
    # choosing the type re-enables Update
    assert "renderLayers(); updateCreate(); };" in HTML


# ---------------------------------------------------------------- #56
def test_reopen_returns_the_projects_own_dates(sandbox):
    legacy = mb.api_project({"config": str(_legacy_project(sandbox))})
    assert (legacy["start"], legacy["end"]) == ("2016-01-01", "2026-06-30")
    r = mb.api_create({"name": "Builder Site", "route": "miaplpy", "track": 19, "direction": "ASCENDING",
                       "start": "2017-03-01", "end": "2026-05-31", "pad_km": 7,
                       "layers": [{"role": "area", "name": "Zone", "geojson": ZONE.geojson()}]})
    d = mb.api_project({"config": r["config"]})
    assert (d["start"], d["end"], d["pad_km"]) == ("2017-03-01", "2026-05-31", 7.0)


def test_builder_page_restores_the_dates_on_open():
    i = HTML.index("$('openSel').onchange")
    handler = HTML[i:HTML.index("\n};", i)]
    for s in ("$('start').value = d.start", "$('end').value = d.end", "$('pad').value = d.pad_km"):
        assert s in handler, s
    # New goes back to the defaults, not the last opened project's dates
    r = HTML.index("function reset(")
    assert "$('end').value = state.defaultEnd" in HTML[r:HTML.index("\n}", r)]


# ---------------------------------------------------------------- #58
def _poly(path: Path, w, s, e, n) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                    "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}]}))
    return path


def _config(tmp_path, report_areas) -> Path:
    ctx = tmp_path / "proj" / "context"
    doc = {"backend": "miaplpy", "project_dir": str(tmp_path / "proj"),
           "bbox": [-87.457, 38.881, -87.203, 39.079], "start_date": "2016-01-01", "end_date": "2026-06-30",
           "track": 165, "orbit_direction": "descending", "report_areas": report_areas,
           "mintpy_deramp": "linear",
           "deramp_exclude_geojson": str(_poly(ctx / "deforming_area.geojson", -87.36, 38.98, -87.33, 39.01)),
           "stack_swaths": "2", "stack_unw_engine": "snaphu2", "stack_unw_cost_mode": "DEFO",
           "stack_virtual_merge": "False", "stack_cleanup_coreg_after_filter": False,
           "miaplpy_phase_linking_method": "sequential_EMI", "miaplpy_ministack_size": 10}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def test_relative_report_polygon_is_refused_at_load(tmp_path):
    pit = _poly(tmp_path / "proj" / "context" / "pit.geojson", -87.35, 38.99, -87.34, 39.0)
    assert load_config(str(_config(tmp_path, ["Full Satellite Imaging:full", f"Pit:{pit}"])))
    # present under project_dir/context, but the report opens it from where
    # the run starts: it passed here and failed at the end of the run
    with pytest.raises(ConfigError, match="relative path"):
        load_config(str(_config(tmp_path, ["Full Satellite Imaging:full", "Pit:context/pit.geojson"])))
    with pytest.raises(ConfigError, match="relative path"):
        cfg = _config(tmp_path, ["Full Satellite Imaging:full"])
        doc = yaml.safe_load(cfg.read_text())
        doc["focus_sites_lalo"] = ["Pit:context/pit.geojson"]
        cfg.write_text(yaml.safe_dump(doc))
        load_config(str(cfg))


# ---------------------------------------------------------------- #59
def _snapshot(ctx: Path) -> dict:
    return {f.name: f.read_bytes() for f in ctx.glob("*.geojson")}


def test_refused_update_leaves_the_context_outlines_as_they_were(tmp_path):
    kw = dict(track=19, direction="ascending", swaths=[2], projects_root=tmp_path / "p", configs_dir=tmp_path / "c")
    # polygons, not boxes: a box report area is written into the config as S,W,N,E
    pond1 = areas.parse_coordinates("POLYGON((-87.69 38.31, -87.66 38.31, -87.66 38.33, -87.69 38.31))")
    pond2 = areas.parse_coordinates("POLYGON((-87.68 38.315, -87.65 38.315, -87.65 38.335, -87.68 38.315))")
    site = areas.box(38.30, -87.70, 38.34, -87.64, "site")
    out = areas.write_project("Pond Site", site, "miaplpy", "2020-01-01", "2024-12-31", deforming=pond1,
                              report_areas=[("Full Satellite Imaging", "full"), ("Site", site), ("Pond", pond1)], **kw)
    ctx = load_config(str(out)).project_dir / "context"
    before, cfg_before = _snapshot(ctx), out.read_text()
    assert {"deforming_area.geojson", "report_pond.geojson", "other_movement.geojson"} <= set(before)
    future = (date.today() + timedelta(days=365)).isoformat()
    # refused by load_config (end date in the future), with a new surface layer
    # that would add a file context/ did not hold
    with pytest.raises(ConfigError):
        areas.write_project("Pond Site", site, "miaplpy", "2020-01-01", future, deforming=pond2, surface=pond2,
                            report_areas=[("Full Satellite Imaging", "full"), ("Site", site), ("Pond", pond2)],
                            overwrite=True, **kw)
    assert out.read_text() == cfg_before
    assert _snapshot(ctx) == before                       # same files, same bytes; no surface_mines.geojson
    # a bad reference spot is refused before anything is written
    with pytest.raises(ValueError, match="latitude"):
        areas.write_project("Pond Site", site, "miaplpy", "2020-01-01", "2024-12-31", deforming=pond2,
                            report_areas=[("Full Satellite Imaging", "full"), ("Site", site), ("Pond", pond2)],
                            reference_lalo="95.0,-87.66", overwrite=True, **kw)
    assert out.read_text() == cfg_before and _snapshot(ctx) == before
