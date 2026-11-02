"""P0.2 (review 8 Oct 2026): the real disk budget on a SPARSE WSL vhdx.

settings.disk_budget_gb used (vhdx LENGTH - ext4 used) + C: free. WSL's vhdx
is sparse: blocks ext4 frees go back to Windows and the length stays, so the
length is not space on C:. The real figure is
    C: free + max(0, vhdx HELD on C: - ext4 used)
(review_2026-10-08/disk/notes.md section 1, real_budget.py): 174 GB on 7 Oct
2026 where the pipeline said 272. Every test here except the ext4 free-space
cap (unchanged behaviour) fails on the code before the fix (checked 8 Oct:
9 of 10, each on the budget figure, the new keys or the missing helper).
Synthetic only: no real vhdx,
nothing under ~/insar_projects or ~/.config/insar read or written.
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from insar_pipeline import doctor, pipeline, procs, settings

DU = namedtuple("usage", "total used free")
OLD_KEYS = {"fs_free", "vhdx_gb", "used_gb", "c_free", "budget", "source", "host_drive"}
NEW_KEYS = {"vhdx_held_gb", "vhdx_headroom_gb", "vhdx_sparse"}

# MEASURED on Kyle's machine, 7 Oct 2026 22:00 (disk/notes.md section 1)
LENGTH, HELD, USED, C_FREE, FS_FREE = 474.8, 377.1, 367.6, 164.5, 658.5


def _fake_wsl(monkeypatch, tmp_path, *, length=LENGTH, held=HELD, used=USED, c_free=C_FREE,
              fs_free=FS_FREE, patch_held=True):
    """WSL root filesystem with its vhdx on a fake drive. The vhdx is a real
    sparse file of `length` (GB, so it costs no disk); its allocation on the
    drive is `held` (patched, since a test cannot hold 377 GB). The state is
    a dict the test can change between calls (a run writing)."""
    drive = tmp_path / "mnt_c"
    vh = drive / "Users" / "u" / "AppData" / "Local" / "wsl" / "x" / "ext4.vhdx"
    vh.parent.mkdir(parents=True, exist_ok=True)
    with open(vh, "wb") as f:
        f.truncate(int(length * 1e9))
    host = str(Path("/", *vh.parts[1:3]))
    st = {"length": length, "held": held, "used": used, "c_free": c_free, "fs_free": fs_free}

    def usage(p):
        if str(p) == host:
            return DU(1000e9, (1000 - st["c_free"]) * 1e9, st["c_free"] * 1e9)
        return DU(1081e9, st["used"] * 1e9, st["fs_free"] * 1e9)
    monkeypatch.setattr(settings, "_is_wsl", lambda: True)
    monkeypatch.setattr(settings, "_mount_kind", lambda p: "root")
    monkeypatch.setattr(settings, "_vhdx_path", lambda: (vh, "test"))
    monkeypatch.setattr(settings, "_usage", usage)
    if patch_held:
        # raising=False: on the old code there is no such helper, the old
        # formula runs on the file's length and the assertions catch it
        monkeypatch.setattr(settings, "_vhdx_sizes",
                            lambda v: (st["length"] * 1e9, st["held"] * 1e9), raising=False)
    return st


# ---------------------------------------------------------------- the formula
def test_sparse_vhdx_budget_is_drive_free_plus_held_minus_used(tmp_path, monkeypatch):
    _fake_wsl(monkeypatch, tmp_path)
    b = settings.disk_budget_gb(tmp_path / "projects" / "not_yet")
    # 164.5 + (377.1 - 367.6) = 174.0, not (474.8 - 367.6) + 164.5 = 271.7
    assert b["budget"] == pytest.approx(174.0, abs=0.15)
    assert b["vhdx_held_gb"] == pytest.approx(HELD, abs=0.05)
    assert b["vhdx_headroom_gb"] == pytest.approx(HELD - USED, abs=0.15)
    assert b["vhdx_sparse"] is True
    assert b["vhdx_gb"] == pytest.approx(LENGTH, abs=0.05)              # still the length, as before
    assert b["c_free"] == pytest.approx(C_FREE) and b["used_gb"] == pytest.approx(USED)
    assert b["budget"] == pytest.approx(b["c_free"] + b["vhdx_headroom_gb"], abs=0.15)
    assert "sparse" in b["source"] and b["host_drive"] in b["source"]   # the drive holding the vhdx


def test_keys_are_kept_and_the_components_added(tmp_path, monkeypatch):
    _fake_wsl(monkeypatch, tmp_path)
    assert set(settings.disk_budget_gb(tmp_path)) >= OLD_KEYS | NEW_KEYS
    monkeypatch.setattr(settings, "_is_wsl", lambda: False)
    b = settings.disk_budget_gb(tmp_path)
    assert set(b) >= OLD_KEYS | NEW_KEYS                                   # same shape off WSL
    assert all(b[k] is None for k in NEW_KEYS) and b["budget"] == pytest.approx(FS_FREE)


def test_unallocated_sparse_vhdx_counts_only_the_drive_free(tmp_path, monkeypatch):
    """No patch of the allocation: the fake vhdx is a real sparse file that
    holds ~nothing, read through st_blocks. Nothing of its length is on C:."""
    _fake_wsl(monkeypatch, tmp_path, patch_held=False, c_free=150.0, used=283.0, fs_free=743.0)
    b = settings.disk_budget_gb(tmp_path)
    assert b["budget"] == pytest.approx(150.0, abs=0.15)                 # old code: 341.8
    assert b["vhdx_held_gb"] == pytest.approx(0.0, abs=0.05) and b["vhdx_sparse"] is True


def test_allocation_is_read_from_the_file_blocks(tmp_path):
    """_vhdx_sizes reads what the file takes on its drive (st_blocks), not
    its length: 3 MB written into a 50 MB sparse file."""
    f = tmp_path / "ext4.vhdx"
    with open(f, "wb") as h:
        h.write(b"\1" * 3_000_000)
        h.truncate(50_000_000)
    length, held = settings._vhdx_sizes(f)
    assert length == 50_000_000
    assert 3_000_000 <= held < 3_200_000                                 # block rounding only
    g = tmp_path / "full.bin"
    g.write_bytes(b"\1" * 1_000_000)
    assert settings._vhdx_sizes(g) == (1_000_000, 1_000_000)              # capped at the length


def test_non_sparse_vhdx_gives_the_earlier_figure(tmp_path, monkeypatch):
    """A fully allocated vhdx (held = length): the same number as before P0.2."""
    _fake_wsl(monkeypatch, tmp_path, length=475.0, held=475.0, used=283.0, c_free=150.0, fs_free=743.0)
    b = settings.disk_budget_gb(tmp_path)
    assert b["budget"] == pytest.approx(342.0, abs=0.15) and b["vhdx_sparse"] is False
    assert "sparse" not in b["source"]


def test_still_capped_by_ext4_free(tmp_path, monkeypatch):
    _fake_wsl(monkeypatch, tmp_path, fs_free=100.0)
    assert settings.disk_budget_gb(tmp_path)["budget"] == pytest.approx(100.0, abs=0.15)


def test_budget_falls_one_gb_per_gb_written(tmp_path, monkeypatch):
    """Writing first fills the blocks the vhdx holds but ext4 does not use,
    then grows the file out of C:. Either way the budget falls by what was
    written. The length-based figure fell 2 GB per GB once the file grew."""
    st = _fake_wsl(monkeypatch, tmp_path)
    b0 = settings.disk_budget_gb(tmp_path)["budget"]
    st["used"] += 5.0                                                     # into the held, unused blocks
    b1 = settings.disk_budget_gb(tmp_path)["budget"]
    assert b0 - b1 == pytest.approx(5.0, abs=0.15)
    grow = st["used"] + 20.0 - st["held"]                                 # 20 GB more: 15.5 GB from C:
    st["used"] += 20.0
    st["held"] += grow
    st["c_free"] -= grow
    b2 = settings.disk_budget_gb(tmp_path)["budget"]
    assert b1 - b2 == pytest.approx(20.0, abs=0.15)
    assert b2 == pytest.approx(st["c_free"], abs=0.15)                   # no headroom left: C: free only


# ---------------------------------------------------------------- what reads it
def test_disk_gate_refuses_a_run_only_the_length_figure_would_pass(tmp_path, monkeypatch):
    _fake_wsl(monkeypatch, tmp_path)
    cfg = SimpleNamespace(project_dir=tmp_path / "projects" / "corcoran_b1")
    args = SimpleNamespace(force=False)
    with pytest.raises(SystemExit, match="disk budget is 174 GB"):
        pipeline.disk_gate(cfg, SimpleNamespace(est_peak_gb=200.0), args)
    out = pipeline.disk_gate(cfg, SimpleNamespace(est_peak_gb=150.0), args)   # fits: passes
    assert out["budget_gb"] == pytest.approx(174.0, abs=0.15) and "sparse" in out["source"]
    assert pipeline.disk_budget(cfg.project_dir.parent)[0] == pytest.approx(174.0, abs=0.15)


def test_watchdog_warns_and_stops_on_the_real_budget(tmp_path, monkeypatch):
    """5 GB unused inside the vhdx + 12 GB on C: = 17 GB, below the 20 GB
    stop. The length-based figure said 115 GB and the run would go on until
    C: filled."""
    _fake_wsl(monkeypatch, tmp_path, used=372.1, c_free=12.0)
    proj = tmp_path / "projects" / "p"
    proj.mkdir(parents=True)
    killed, interrupted = [], []
    monkeypatch.setattr(procs, "kill_children", lambda *a, **k: killed.append(1) or 0)
    monkeypatch.setattr(pipeline._thread, "interrupt_main", lambda: interrupted.append(1))
    wd = pipeline.DiskWatchdog(SimpleNamespace(project_dir=proj))
    wd.check()
    assert wd.warned and wd.tripped is None                               # one low sample: warn only
    wd.check()
    assert wd.tripped and "17 GB left" in wd.tripped and killed and interrupted


def test_doctor_shows_the_components_and_warns_on_the_real_budget(tmp_path, monkeypatch):
    """A 900 GB-long sparse vhdx: the length-based figure (697 GB) passed the
    350 GB phase-linked check; the real 174 GB does not."""
    _fake_wsl(monkeypatch, tmp_path, length=900.0)
    lines = doctor.disk_checks(tmp_path)
    head = lines[0][1]
    for s in ("174 GB real budget", "164 GB free +", "10 GB unused", "377 GB", "368 GB", "sparse", "900 GB"):
        assert s in head, (s, head)
    room = next(x for x in lines if "phase-linked run" in x[1])
    assert room[0] is False and room[3] is True and "174 GB available" in room[1]
