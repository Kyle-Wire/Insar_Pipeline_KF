"""The stack-route disk plan counts real merged SLCs (22 Oct 2026). With
stack_virtual_merge False, run_12 writes a full-resolution SLC per date beside
its coregistered bursts; the plan left them out and printed 303 GB for a Site D
full-record run that needed about 475 GB at the merge."""
from types import SimpleNamespace

import pytest

from insar_pipeline import data_acquisition as da, stack_backend as sb
from insar_pipeline.config import PipelineConfig


def _plan(tmp_path, monkeypatch, virtual, backend="stack", n_dates=100, **over):
    cfg = PipelineConfig(project_dir=tmp_path / "p", backend=backend, start_date="2020-01-01",
                         end_date="2024-01-01", bbox=[-87.5, 38.2, -87.3, 38.4], track=19,
                         stack_virtual_merge=virtual, stack_range_looks=4, stack_azimuth_looks=1,
                         stack_num_connections=1, **over)
    scenes = [SimpleNamespace(properties={"startTime": f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}T00:00:00"})
              for i in range(n_dates)]
    monkeypatch.setattr(da, "search_scenes", lambda c: scenes)
    monkeypatch.setattr(sb, "_bursts_per_date", lambda c, n: (3, "test"))
    monkeypatch.setattr(sb, "_crop_fraction", lambda c, b: (0.25, "test"))
    monkeypatch.setattr(sb, "_disk_budget", lambda p: (1000.0, "test"))
    monkeypatch.setattr(sb, "_allocated_gb", lambda p: 0.0)
    return sb.plan_stack(cfg)


def test_a_real_merge_on_the_stack_route_is_counted(tmp_path, monkeypatch):
    virtual = _plan(tmp_path, monkeypatch, "True")
    real = _plan(tmp_path, monkeypatch, "False")
    assert real.est_peak_gb - virtual.est_peak_gb == pytest.approx(0.84 * 100, abs=0.2)
    note = next(n for n in real.notes if n.startswith("Disk by moment"))
    assert "end of the merge" in note and "real merged SLCs ~84 GB" in note
    assert "end of the merge" not in next(n for n in virtual.notes if n.startswith("Disk by moment"))


def test_the_phase_linked_route_is_not_counted_twice(tmp_path, monkeypatch):
    # the miaplpy merge swaps each date's bursts for its merged copy; its own
    # block already sizes the merged SLCs
    p = _plan(tmp_path, monkeypatch, "False", backend="miaplpy", miaplpy_sparse_merged_slc=False)
    note = next(n for n in p.notes if n.startswith("Disk by moment"))
    assert "end of the merge" not in note
    assert p.est_merged_gb == pytest.approx(0.84 * 100, abs=0.2)
    # 11 Nov 2026: kept to the crop (sparse), counted once and smaller
    s = _plan(tmp_path, monkeypatch, "False", backend="miaplpy")
    assert "end of the merge" not in next(n for n in s.notes if n.startswith("Disk by moment"))
    assert s.est_merged_gb < p.est_merged_gb
