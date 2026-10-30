"""13 Oct 2026: 'keep results only' runs by itself once a project is finished
(FINAL package built, or an experiment run ended), never otherwise."""
from types import SimpleNamespace

from insar_pipeline import package, pipeline, storage


def test_off_switch_and_never_raises(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(storage, "keep_results_only", lambda name, apply=False: calls.append(name) or
                        {"applied": True, "freed": 5e9, "refused": [], "failed": []})
    cfg = SimpleNamespace(project_dir=tmp_path, auto_keep_results=False)
    assert storage.auto_keep_results(cfg, "x") is None and calls == []
    cfg.auto_keep_results = True
    assert storage.auto_keep_results(cfg, "x")["applied"] and calls == [str(tmp_path)]

    def boom(name, apply=False):
        raise OSError("disk gone")
    monkeypatch.setattr(storage, "keep_results_only", boom)
    assert storage.auto_keep_results(cfg, "x") is None                 # logged, not raised


def test_refusal_is_reported_not_forced(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "keep_results_only", lambda name, apply=False:
                        {"applied": False, "refused": ["The project is not finished"], "failed": []})
    plan = storage.auto_keep_results(SimpleNamespace(project_dir=tmp_path), "x")
    assert plan["refused"] and not plan["applied"]


def test_final_package_triggers_it_and_a_draft_does_not(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(storage, "auto_keep_results", lambda cfg, why: seen.append(why))
    cfg = SimpleNamespace(project_dir=tmp_path)
    monkeypatch.setattr(package, "_build", lambda cfg, force_draft: tmp_path / "releases" / "20261013_1200")
    package.build(cfg)
    assert len(seen) == 1 and "FINAL" in seen[0]
    monkeypatch.setattr(package, "_build", lambda cfg, force_draft: tmp_path / "releases" / "20261013_1200_DRAFT")
    package.build(cfg, force_draft=True)
    assert len(seen) == 1


def test_experiment_run_triggers_it_only_when_done_and_not_when_sizing(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(storage, "auto_keep_results", lambda cfg, why: seen.append(why))
    monkeypatch.setattr(pipeline, "is_experiment", lambda cfg: True)
    cfg = SimpleNamespace(project_dir=tmp_path)
    monkeypatch.setattr(pipeline, "_run_from", lambda cfg, start, **o: 0)
    pipeline.run_from(cfg, "full", kind="plan", plan_only=True)
    assert seen == []
    pipeline.run_from(cfg, "products")
    assert len(seen) == 1
    monkeypatch.setattr(pipeline, "_run_from", lambda cfg, start, **o: 1)     # failed run
    pipeline.run_from(cfg, "products")
    assert len(seen) == 1
    monkeypatch.setattr(pipeline, "_run_from", lambda cfg, start, **o: 0)
    monkeypatch.setattr(pipeline, "is_experiment", lambda cfg: False)         # client project: package does it
    pipeline.run_from(cfg, "products")
    assert len(seen) == 1
