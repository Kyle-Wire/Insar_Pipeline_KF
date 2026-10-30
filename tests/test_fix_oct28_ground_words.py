"""The comparison ground in words (28 Oct 2026, Kyle).

control_ground_label's default, "farmland", was written for the Indiana
coalfield. The first combined site's comparison ground (significance.far_field_mask, the
report's ctx.far_wide) is 53% built-up, 44% grassland and 0% cropland, yet
its client report said "farmland" throughout. The words now come from the
measured land cover (landcover.comparison_ground) unless someone set the
label: "farmland" only where cropland dominates, else "ground (mostly
built-up and grassland)". The report, the KMZ, the raster tags, the figures
and the QA prose say the same words; significance's own sentences say
"ordinary ground" at the source.

Synthetic projects in tmp_path only; no network.
"""
import json
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L5 as L5                                          # noqa: E402  (the client-report project builder)
import test_fix_L8 as L8                                          # noqa: E402  (the QA project builder)

from insar_pipeline import client_kmz as ck                       # noqa: E402
from insar_pipeline import client_report as cr                    # noqa: E402
from insar_pipeline import figures                                # noqa: E402
from insar_pipeline import landcover                              # noqa: E402
from insar_pipeline import quality_assessment as qa               # noqa: E402
from insar_pipeline import report_insights as ri                  # noqa: E402
from insar_pipeline import significance as sg                     # noqa: E402

NORA = "ground (mostly built-up and grassland)"
no_network = L8.no_network                                        # autouse: the QA never fetches OSM water
FARMLAND_WORD = re.compile(r"\bfarmland\b", re.I)


@pytest.fixture(autouse=True)
def _clear_cache():
    landcover._GRID_CACHE.clear()
    yield
    landcover._GRID_CACHE.clear()


def _write_wc(path, classes, x0, y0, dx, dy):
    """A WorldCover-coded raster exactly on a product grid."""
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", width=classes.shape[1], height=classes.shape[0], count=1,
                       dtype="uint8", crs="EPSG:4326", nodata=0,
                       transform=from_origin(x0, y0, dx, -dy)) as d:
        d.write(classes.astype(np.uint8), 1)
        d.update_tags(scheme="worldcover")
    return str(path)


def _town(shape):
    """The first combined site's mix: built-up on the left 55%, grassland on the rest."""
    cls = np.full(shape, 30)
    cls[:, : int(0.55 * shape[1])] = 50
    return cls


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


# ------------------------------------------------------------------ the rule
def test_words_from_the_measured_cover():
    nora = {"cover_percent": {"built-up": 53.2, "grassland": 44.0, "bare ground": 2.7, "cropland": 0.0}}
    assert landcover.ground_label_from("", nora) == NORA
    # the same ground measured twice (QA's pass, the report's) in the other
    # order of shares: the same words
    again = {"cover_percent": {"grassland": 48.7, "built-up": 48.3, "bare ground": 2.8}}
    assert landcover.ground_label_from("", again) == NORA
    # cropland dominating keeps "farmland" (Site A: 87-93% cropland)
    assert landcover.ground_label_from("", {"cover_percent": {"cropland": 86.8, "grassland": 10.0}}) == "farmland"
    # NLCD pasture counts as farmland
    assert landcover.ground_label_from("", {"cover_percent": {"cropland": 30.0, "pasture": 25.0,
                                                              "developed": 45.0}}) == "farmland"
    # a cover under GROUND_NAMED_SHARE is not named
    assert landcover.ground_label_from("", {"cover_percent": {"grassland": 70.0, "shrubland": 15.0,
                                                              "cropland": 15.0}}) == "ground (mostly grassland)"
    # no map: plain words, never a guess
    assert landcover.ground_label_from("", None) == "ground"
    # a label someone set wins over any measurement
    assert landcover.ground_label_from("open rangeland", nora) == "open rangeland"


def test_only_a_label_someone_set_counts_as_set():
    assert landcover.configured_ground_label(SimpleNamespace(control_ground_label="farmland")) == ""
    assert landcover.configured_ground_label(SimpleNamespace(control_ground_label=" Farmland ")) == ""
    assert landcover.configured_ground_label(SimpleNamespace(control_ground_label="")) == ""
    assert landcover.configured_ground_label(SimpleNamespace()) == ""
    assert landcover.configured_ground_label(SimpleNamespace(control_ground_label="pasture")) == "pasture"


def test_comparison_ground_measures_the_mask_it_is_given(tmp_path):
    geo = {"X_FIRST": -105.0, "Y_FIRST": 40.0, "X_STEP": 0.001, "Y_STEP": -0.001, "LENGTH": 50, "WIDTH": 80}
    cls = np.full((50, 80), 40)                                   # cropland ...
    cls[:, 40:] = 50                                              # ... and a town
    p = _write_wc(tmp_path / "lc.tif", cls, -105.0, 40.0, 0.001, -0.001)
    cfg = SimpleNamespace(landcover_raster=p, control_ground_label="farmland")
    town = np.zeros((50, 80), bool); town[:, 45:] = True
    fields = np.zeros((50, 80), bool); fields[:, :35] = True
    g = landcover.comparison_ground(cfg, geo, town)
    assert g["label"] == "ground (mostly built-up)" and g["source"] == "measured"
    assert g["cover_percent"] == {"built-up": 100.0}
    assert landcover.comparison_ground(cfg, geo, fields)["label"] == "farmland"
    # set: not measured at all
    own = landcover.comparison_ground(SimpleNamespace(landcover_raster=p, control_ground_label="pasture"), geo, town)
    assert own == {"label": "pasture", "source": "config", "cover_percent": None, "scheme": None}
    # no map
    none = landcover.comparison_ground(SimpleNamespace(landcover_raster="", control_ground_label="farmland"), geo, town)
    assert none["label"] == "ground" and none["source"] == "unmeasured"


def test_say_ground_rewords_sentences_not_identifiers_or_names():
    rec = {"verdict": "within the spread of ordinary farmland",
           "basis": "Farmland copies of this footprint; farmland_spread_1sigma_mm_yr",
           "reference_mode": "farmland", "name": "Farmland North", "nested": [{"text": "the surrounding farmland"}],
           "path": "a/farmland/b.tif"}
    out = ri.say_ground(rec, NORA)
    assert out["verdict"] == "within the spread of ordinary ground"
    assert out["basis"] == "Ground copies of this footprint; farmland_spread_1sigma_mm_yr"
    assert out["reference_mode"] == "farmland"                     # a value, not a sentence
    assert out["name"] == "Farmland North"                         # a name someone gave
    assert out["nested"][0]["text"] == "the surrounding ground"
    assert ri.say_ground(rec, "farmland") == rec                   # farmland stays farmland
    assert ri.say_ground("over farmland here", "open rangeland") == "over open rangeland here"
    assert ri.ground_noun(NORA) == "ground" and ri.ground_noun("pasture") == "pasture"


def test_significance_says_ordinary_ground_at_the_source():
    """assess's not-testable reasons and verdicts reach the QA, the report,
    the workbook and the combined view verbatim: they never say farmland."""
    field = np.zeros((40, 40))
    far = np.ones((40, 40), bool)
    tiny = np.zeros((40, 40), bool); tiny[0, :3] = True
    r = sg.assess(field, far, tiny)
    assert r["testable"] is False and "ordinary ground" in r["not_testable_reason"]
    huge = np.zeros((40, 40), bool); huge[:, :30] = True
    r2 = sg.assess(np.random.default_rng(1).normal(0, 1, (40, 40)), far & ~huge, huge)
    assert r2["testable"] is False and not FARMLAND_WORD.search(r2["not_testable_reason"])


# ------------------------------------------------------------------ the QA and the figures
def _qa_project(tmp_path, cls=None, **over):
    cfg = L8.build(tmp_path, **over)
    if cls is not None:
        cfg.landcover_raster = _write_wc(tmp_path / "lc.tif", cls, L8.X0, L8.Y0, L8.DX, L8.DY)
    return cfg


def test_qa_words_its_record_with_the_measured_ground(tmp_path):
    r = qa.evaluate(_qa_project(tmp_path, _town((L8.NY, L8.NX))))
    cg = r["comparison_ground"]
    assert cg["source"] == "measured" and cg["words"] == NORA
    # every sentence but the two that name the farmland site PROFILE (the
    # send rule's rule line and the mismatch note below)
    profile = ("the site profile is farmland", "farmland-expected flags")
    prose = [s for s in _strings({k: v for k, v in r.items() if k != "comparison_ground"})
             if " " in s and not any(p in s for p in profile)]
    assert prose and not [s for s in prose if FARMLAND_WORD.search(s)]
    # the farmland profile's expectations stay; the mismatch is a note, not a hold
    assert all("on_farmland_expected" in c for c in r["checks"].values())
    note = [n for n in r["send_rule"]["notes"] if n.startswith("the site profile is farmland")]
    assert note and NORA in note[0]
    assert figures.ground_words(SimpleNamespace(control_ground_label="farmland"), r) == NORA


def test_qa_keeps_farmland_on_cropland_and_a_set_label_wins(tmp_path):
    crops = np.full((L8.NY, L8.NX), 40)
    r = qa.evaluate(_qa_project(tmp_path / "a", crops))
    assert r["comparison_ground"]["words"] == "farmland"
    assert not [n for n in r["send_rule"]["notes"] if n.startswith("the site profile is farmland")]
    r2 = qa.evaluate(_qa_project(tmp_path / "b", _town((L8.NY, L8.NX)), control_ground_label="pasture"))
    assert r2["comparison_ground"]["words"] == "pasture"
    assert not [n for n in r2["send_rule"]["notes"] if n.startswith("the site profile is farmland")]
    assert figures.ground_words(SimpleNamespace(control_ground_label="pasture"), r2) == "pasture"
    # an older record (no comparison_ground): the config's label, as before
    assert figures.ground_words(SimpleNamespace(control_ground_label="farmland"), {}) == "farmland"


# ------------------------------------------------------------------ the report, the KMZ, the rasters
@pytest.fixture(scope="module")
def town_report(tmp_path_factory):
    """L5's synthetic report project on the first combined site's mix of land cover: its
    client report and KMZ built once."""
    root = tmp_path_factory.mktemp("ground_words")
    mp = pytest.MonkeyPatch()
    mp.setattr(ri.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline (test)")))
    cfg = L5.make_project(root)
    cfg.landcover_raster = _write_wc(root / "lc.tif", _town((L5.NY, L5.NX)), L5.X0, L5.Y0, L5.DX, L5.DY)
    page = cr.build(cfg)
    D = cfg.project_dir / "deliverables"
    rep = json.loads((D / "client_report.json").read_text())
    kmz = ck.build(cfg, out_dir=root / "preview")
    mp.undo()
    return SimpleNamespace(cfg=cfg, rep=rep, html=page.read_text(), D=D,
                           kml=zipfile.ZipFile(kmz).read("doc.kml").decode())


def test_report_measures_its_own_comparison_ground(town_report):
    rep = town_report.rep
    assert rep["control_ground_label"] == NORA
    assert rep["comparison_ground"]["source"] == "measured"
    assert set(rep["comparison_ground"]["cover_percent"]) <= {"built-up", "grassland"}
    for f in rep["features"]:
        assert f["control_ground_label"] == NORA
        assert f"relative to the surrounding {NORA}" in f["text"]


def test_no_printer_says_farmland_on_ground_that_is_not(town_report):
    rep = town_report.rep
    text = L5._text(town_report.html)
    assert not FARMLAND_WORD.search(text), FARMLAND_WORD.findall(text)
    assert f"Ordinary {NORA} around the site" in text              # the glossary names the cover once
    bad = [s for s in _strings(rep) if " " in s and FARMLAND_WORD.search(s)]
    assert not bad, bad[:3]
    assert not FARMLAND_WORD.search(town_report.kml)
    assert NORA in town_report.kml


def test_report_rasters_are_tagged_with_the_same_words(town_report):
    import rasterio
    unit = "in"
    with rasterio.open(town_report.D / f"vertical_movement_{unit}_per_yr.tif") as d:
        tags = d.tags()
        desc = d.descriptions[0] or ""
    assert tags["RELATIVE_TO"].startswith(f"the surrounding {NORA} of the same coherence class")
    assert not [v for v in tags.values() if FARMLAND_WORD.search(v)] and not FARMLAND_WORD.search(desc)
    with rasterio.open(town_report.D / f"vertical_movement_uncertainty_1sigma_{unit}_per_yr.tif") as d:
        assert not FARMLAND_WORD.search(d.descriptions[0] or "")
    with rasterio.open(town_report.D / "rate_signal_to_noise.tif") as d:
        assert NORA in d.tags()["RELATIVE_TO"]
