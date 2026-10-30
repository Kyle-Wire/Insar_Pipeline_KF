"""
Phase 0 finishing pass (8 Oct 2026, review_2026-10-08/PLAN.md section 3).

P0.2 made the disk budget count what the sparse WSL vhdx HOLDS on C:, not its
length. The dashboard's disk panel still printed the length ("vhdx 475 GB"),
which on a sparse file is not space on C: and does not add up to the budget
shown beside it. P0.3 found DISP-S1 frames outside North America (F39360 over
Sydney/Appin); the cross-check and the dashboard still told the user DISP-S1
covers North America only.

Each test here fails on the code the groups left.
"""
from pathlib import Path

from insar_pipeline import dashboard

PKG = Path(__file__).resolve().parents[1]


def _plain(text) -> str:
    return text.plain if hasattr(text, "plain") else str(text)


def test_disk_panel_shows_the_parts_of_the_real_budget_not_the_vhdx_length():
    # 8 Oct 2026, this machine: C: free 164.1, held 377.1, used 367.7, length 474.8
    r = {"free_gb": 658.5, "budget_gb": 173.5, "c_free_gb": 164.1, "vhdx_gb": 474.8,
         "vhdx_headroom_gb": 9.5}
    out = _plain(dashboard.Resources.render(r))
    assert "real budget 174 GB" in out
    assert "C: 164 GB free + 10 GB unused in the WSL disk" in out      # 164 + 10 = 174, as shown
    assert "vhdx 475" not in out                                       # the length is not space on C:


def test_disk_panel_without_the_new_parts_still_renders():
    # a settings module that returns only the old keys (or a non-WSL figure)
    r = {"free_gb": 500.0, "budget_gb": 400.0, "c_free_gb": 150.0, "vhdx_gb": 475.0}
    out = _plain(dashboard.Resources.render(r))
    assert "real budget 400 GB" in out and "C: 150 GB free" in out


def test_disk_sample_carries_the_vhdx_headroom(monkeypatch, tmp_path):
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "disk_budget_gb", lambda p: {
        "fs_free": 658.5, "vhdx_gb": 474.8, "used_gb": 367.7, "c_free": 164.1, "budget": 173.5,
        "source": "WSL: test", "host_drive": "C:", "vhdx_held_gb": 377.1,
        "vhdx_headroom_gb": 9.5, "vhdx_sparse": True})
    r = dashboard.Resources().sample(str(tmp_path))
    assert r["budget_gb"] == 173.5 and r["c_free_gb"] == 164.1
    assert r["vhdx_headroom_gb"] == 9.5


def test_no_disp_frame_messages_do_not_say_north_america_only():
    cross = (PKG / "crosscheck.py").read_text()
    dash = (PKG / "dashboard.py").read_text()
    assert "(North America from mid-2016 only)" not in cross
    assert "frames elsewhere" in cross and "F39360" in cross
    assert "DISP-S1 covers North America from mid-2016" not in dash
    assert "with a few frames elsewhere" in dash


def test_no_module_describes_the_budget_as_vhdx_length_headroom():
    """Docstrings that told a reader the budget is "vhdx headroom + C: free"
    (the length-based formula P0.2 replaced)."""
    for name in ("advice.py", "stack_backend.py", "dashboard.py", "pipeline.py", "settings.py"):
        text = (PKG / name).read_text()
        assert "vhdx headroom + C: free" not in text, name
        assert "vhdx\n    headroom + the Windows drive" not in text, name
