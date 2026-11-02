"""Second review, 7 Oct 2026: the installer and the documents.

#60  Re-running install.sh (the documented update and rollback both use
     --no-credentials) reset a projects folder moved with
     `insar doctor --projects` back to ~/insar_projects: its default ignored
     the saved settings. Step 7 is run here in bash with a temporary HOME and
     settings file; nothing real is installed or touched.
#61-#66 and the round-2 handoffs: the documents no longer quote withdrawn
     figures or methods (median centres, the uplift injection run, a
     delivered disturbed-ground layer) and name what the code now does.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def _read(doc: str) -> str:
    # HANDOFF.md stays on the developer's machine (16 Oct 2026): skip on a fresh clone
    from client_files import need
    return need(PKG / doc, "kept on the developer's machine, not in the repository").read_text()


# ------------------------------------------------------------------ #60 install.sh step 7
def _step7() -> str:
    """install.sh step 7, from its banner to the settings save (the part that
    picks and saves the projects folder), without the `insar` link."""
    lines = (PKG / "install" / "install.sh").read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith('say "7/8'))
    end = next(i for i, ln in enumerate(lines) if i > start and ln.startswith('mkdir -p "$HOME/.local/bin"'))
    return "\n".join(lines[start:end])


def _run_step7(tmp_path: Path, *, saved: str | None, ask: bool = False, env_projects: str | None = None,
               answer: str = "\n") -> dict:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    sfile = tmp_path / "settings.json"
    if saved is not None:
        sfile.write_text(json.dumps({"projects_root": saved, "firm": "Test firm"}))
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    if not (bindir / "python").exists():
        (bindir / "python").symlink_to(sys.executable)
    env = {k: v for k, v in os.environ.items() if k != "INSAR_PROJECTS"}
    env.update(HOME=str(home), INSAR_SETTINGS=str(sfile), PATH=f"{bindir}:{env.get('PATH', '')}",
               PYTHONPATH=str(PKG.parent))
    if env_projects is not None:
        env["INSAR_PROJECTS"] = env_projects
    script = "set -euo pipefail\nsay() { :; }\nASK_CREDS=%d\n%s\n" % (1 if ask else 0, _step7())
    r = subprocess.run(["bash", "-c", script], input=answer, capture_output=True, text=True, env=env,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(sfile.read_text())


def test_update_without_credentials_keeps_a_moved_projects_folder(tmp_path):
    """The update / rollback route (install.sh --no-credentials) keeps the
    folder saved by `insar doctor --projects`; the old default reset it to
    ~/insar_projects."""
    moved = str(tmp_path / "data" / "insar_projects")
    s = _run_step7(tmp_path, saved=moved)
    assert s["projects_root"] == moved
    assert s["firm"] == "Test firm"            # other settings kept


def test_enter_at_the_prompt_keeps_a_moved_projects_folder(tmp_path):
    moved = str(tmp_path / "data" / "insar_projects")
    assert _run_step7(tmp_path, saved=moved, ask=True)["projects_root"] == moved


def test_first_install_still_defaults_to_home(tmp_path):
    s = _run_step7(tmp_path, saved=None)
    assert s["projects_root"] == str(tmp_path / "home" / "insar_projects")


def test_insar_projects_still_wins(tmp_path):
    env_dir = str(tmp_path / "env_projects")
    s = _run_step7(tmp_path, saved=str(tmp_path / "saved"), env_projects=env_dir)
    assert s["projects_root"] == env_dir


def test_install_readme_says_the_folder_in_use_is_kept():
    t = _read("install/README.md")
    assert "insar doctor --projects" in t and "folder already in use" in t


# ------------------------------------------------------------------ #61-#66, handoffs
def test_handoff_no_longer_quotes_the_withdrawn_site_a_figures():
    t = _read("HANDOFF.md")
    sec = t[t.index("### Numbers that move"):t.index("### Deferred, partial")]
    for old in ("−78", "Keep the second farmland pass on?"):
        assert old not in sec and old not in t[t.index("### Open questions"):t.index("### Commands for Kyle")], old
    assert "+0.20" in sec and "−5.64" in sec and "−67" in sec
    # E13 (trust review 7 Oct 2026, findings #7 and #20): −0.89 / −6.73 /
    # −1.92 are the current anchored figures, not withdrawn ones; the section
    # must compare like with like and say the second pass lowers every
    # Site A figure by about 1.05 mm/yr
    assert "+1.26 → +0.20" in sec and "+0.17 → −0.89" in sec and "−5.67 → −6.73" in sec
    assert "−1.09 →\n  −1.92" in sec or "−1.09 → −1.92" in sec
    assert "about 1.05 mm/yr" in sec
    for old in ("−0.89", "−6.73", "−1.92"):
        assert old not in t[t.index("### Open questions"):t.index("### Commands for Kyle")], old


def test_docs_say_class_centres_are_means():
    assert "median class centres" not in _read("TECHNICAL_REFERENCE.md")
    assert "Median coherence-class centres" not in _read("HANDOFF.md")
    assert "mean class centres" in _read("TECHNICAL_REFERENCE.md")


def test_disturbed_ground_layer_is_not_called_delivered():
    t = _read("HANDOFF.md")
    assert "disturbed-ground layers are built, delivered" not in t
    assert "Built and delivered as a diagnostic only" not in t
    r = _read("TECHNICAL_REFERENCE.md")                            # moved from README.md, 2 Nov 2026
    i = r.index("disturbed_ground_runs.tif")
    assert "usable" in r[i:i + 200]


def test_handoff_lists_are_current():
    t = _read("HANDOFF.md")
    assert "793 passed" not in t
    assert "Both projects' quality_assessment.json predate the send rule" not in t
    assert "**A pyflakes gate test** (tests-14): not yet a test" not in t


def test_user_guide_lists_the_identity_and_gnss_tie_blockers():
    t = _read("USER_GUIDE.md")
    i = t.index("Hard blockers")
    block = t[i:i + 3000]
    assert "--preparer" in block and "GNSS tie" in block


def test_site_b_placement_count_is_one_figure():
    t = _read("USER_GUIDE.md")
    assert "about 4 times" not in t and "3 to 4 times" not in t


def test_injection_figures_are_the_subsidence_rerun():
    for doc in ("HANDOFF.md", "INSAR_RUNBOOK.md"):
        t = _read(doc)
        assert "20 mm/yr 4 of 6" not in t, doc
        assert "injection" not in t or "2 of 6" in t, doc


def test_stage_order_puts_the_checks_before_the_qa():
    r = _read("TECHNICAL_REFERENCE.md")
    assert "injection-recovery test after the QA" not in r
    assert "before the quality assessment" in r


def test_handoff_has_the_second_review_section():
    t = _read("HANDOFF.md")
    i, j = t.index("### 6 Oct: outcome check"), t.index("### 7 Oct: second review")
    assert i < j < t.index("## What this is")
    sec = t[j:t.index("## What this is")]
    for s in ("backups/pre_fixes_20261007/", "tests/test_fix_oct7_", "Needs Kyle", "#10", "#23"):
        assert s in sec, s
