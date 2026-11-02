"""Decisions D14 and D18 (7 Oct 2026, made at Kyle's request).

D14: release tarballs are internal to the operating organisation. Since 2 Nov 2026 they hold
no client data (project configs left on 28 Oct; the machine's local-only tests, the golden
numbers and client report areas in context/ stay out too: readiness review BLD-06).
RELEASE.txt and install/README.md say "internal: do not distribute outside the operating
organisation".

D18: the Site A configs record where their mine layer came from
(mine_layer_sources, in areas.write_project's "<kind label>: source, date"
form). Until 7 Oct neither config had one, so the report credited no mine
layer and said "when they were mapped is not recorded" about the CMIS
outlines, although the package records the source (the geojson's _source
header, context/SOURCE.txt) and the download date (21 Sep 2026).
"""
from __future__ import annotations

import re
import subprocess
import tarfile
from pathlib import Path

import pytest
from client_files import need                                    # client data: skips on a fresh clone

from insar_pipeline.config import load_config

PKG = Path(__file__).resolve().parents[1]

def _load_cfg(name):
    """The package config; the DISP one points at files of the site_a_pl
    project (its grid, geometry, baselines), so it is skipped when that
    project is not on this machine (deleted or archived) - 13 Oct 2026."""
    from insar_pipeline.config import ConfigError
    try:
        return load_config(str(need(PKG / "configs" / name)))
    except ConfigError as exc:
        if "_disp" in name and "does not exist" in str(exc):
            pytest.skip(f"{name} needs the site_a_pl project on disk: {str(exc).splitlines()[-1].strip()}")
        raise

INTERNAL = "internal: do not distribute outside the operating organisation"


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# ------------------------------------------------------------------ D14
def test_release_txt_says_the_release_is_internal(tmp_path):
    out = subprocess.run(["bash", str(PKG / "install" / "make_release.sh"), str(tmp_path)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    tars = list(tmp_path.glob("insar_pipeline_*.tar.gz"))
    assert len(tars) == 1, out.stdout
    with tarfile.open(tars[0]) as tf:
        rel = tf.extractfile("insar_pipeline/RELEASE.txt").read().decode()
        names = tf.getnames()
    assert INTERNAL in rel.splitlines()[1], rel
    assert INTERNAL in out.stdout
    # no project config ships (28 Oct 2026: they left the repository; projects
    # keep theirs in the projects folder's configs/)
    assert not [n for n in names if n.startswith("insar_pipeline/configs/") and n.endswith((".yaml", ".yml"))]
    # every page the code serves ships (16 Oct 2026: results_viewer.html did not,
    # so the results map failed on a coworker's install)
    from insar_pipeline import map_builder, results_viewer
    for page in (map_builder.PAGE, results_viewer.PAGE):
        assert f"insar_pipeline/{page}" in names, page
    # and every file the results viewer's page loads (19 Oct 2026: its scripts,
    # styles and vendored libraries moved out of the page and off the CDN)
    for rel, f in results_viewer.ASSETS.items():
        assert f"insar_pipeline/{f.relative_to(results_viewer.PKG).as_posix()}" in names, rel
    # and the app's (24 Oct 2026): its page, every file it loads, the Windows setup and
    # the icon's launcher, with the setup also beside the release file
    from insar_pipeline import app_server
    assert f"insar_pipeline/app_assets/{app_server.PAGE}" in names
    for rel, f in app_server.ASSETS.items():
        assert f"insar_pipeline/{f.relative_to(app_server.PKG).as_posix()}" in names, rel
    for f in ("setup.ps1", "Setup InSAR Pipeline.cmd", "InSAR Pipeline.vbs", "insar.ico"):
        assert f"insar_pipeline/install/windows/{f}" in names, f
    assert (tmp_path / "setup.ps1").exists() and (tmp_path / "Setup InSAR Pipeline.cmd").exists()
    # its checksum beside it, in sha256sum's format with the file name only, so a
    # copy checks with `sha256sum -c` (16 Oct 2026, readiness review BLD-05)
    import hashlib
    sums = Path(str(tars[0]) + ".sha256")
    assert sums.exists(), out.stdout
    assert sums.read_text() == f"{hashlib.sha256(tars[0].read_bytes()).hexdigest()}  {tars[0].name}\n"
    assert hashlib.sha256(tars[0].read_bytes()).hexdigest() in out.stdout
    chk = subprocess.run(["sha256sum", "-c", sums.name], cwd=tmp_path, capture_output=True, text=True)
    assert chk.returncode == 0 and ": OK" in chk.stdout, chk.stdout + chk.stderr


def test_install_readme_says_the_release_is_internal():
    t = _flat((PKG / "install" / "README.md").read_text())
    assert INTERNAL in t
    head = t[:t.index("## 1.")]
    assert INTERNAL in head and "RELEASE.txt" in head


