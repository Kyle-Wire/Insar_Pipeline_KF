"""stack_unw_ncorrlooks (22 Oct 2026): an adaptively filtered stack can tell
snaphu how many looks are behind its coherence; unset, nothing changes."""
from types import SimpleNamespace

from insar_pipeline import unwrap_crop as uc


def _cfg(n):
    return SimpleNamespace(stack_unw_engine="snaphu2", stack_unw_cost_mode="DEFO", stack_unw_max_components=64,
                           stack_unw_min_component_fraction=0.001, stack_unw_ncorrlooks=n, stack_unw_snaphu_bin="")


def test_unset_keeps_the_engine_keys_unchanged(monkeypatch):
    monkeypatch.setattr(uc, "find_snaphu", lambda b: "/bin/snaphu")
    keys = uc.engine_lines(_cfg(0.0))
    assert "ncorrlooks" not in keys
    keys = uc.engine_lines(_cfg(30))
    assert keys["ncorrlooks"] == "30.0"
    text = uc.apply_engine("defomax : 4\nncorrlooks : 12\n", _cfg(30))
    assert text.count("ncorrlooks") == 1 and "ncorrlooks : 30.0" in text
