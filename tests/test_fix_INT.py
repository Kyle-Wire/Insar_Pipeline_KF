"""Final integration (5-6 Oct 2026 review): the pyflakes gate (tests-14) and
golden numbers on the real records (tests-1).

The golden tests rebuild the client report from the real products (about
1-2.5 min per project on 6 Oct 2026; since the 7 Oct follow-ups, 199
simulated copies and the leave-candidate-out re-tests (F1, F3), about 6 min
on site_b and 10-12 on site_a_pl, peak RSS 0.9 and 1.5 GB, measured with
/usr/bin/time beside other jobs), so they run only with INSAR_GOLDEN=1:

    INSAR_GOLDEN=1 python -m pytest insar_pipeline/tests/test_fix_INT.py -q

and skip when a project folder is missing (a coworker's machine). The pinned
values are the fixed code's output on 6 Oct 2026; a change in them must be
deliberate and logged in HANDOFF.md with its cause
(review_2026-10-05/lanes/before_after.md has the before/after of each).
"""
import json
import os
import subprocess
import sys
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
GOLDEN = os.environ.get("INSAR_GOLDEN") == "1"


def test_pyflakes_is_clean_on_the_package():
    """tests-14: a noisy linter hides the next real hit (an undefined name).
    Intentional import checks carry '# noqa'."""
    files = [str(p) for p in sorted(PKG.glob("*.py")) + sorted((PKG / "scripts").glob("*.py"))
             + sorted((PKG / "tests").glob("*.py"))]
    r = subprocess.run([sys.executable, "-m", "pyflakes", *files], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _preview(cfg_path: str, tmp_path: Path) -> tuple[dict, dict]:
    """(client_report.json's content, {area: the mean of the report's vertical
    raster over the area's measured cells, mm/yr}) - the record and the
    raster written from it into tmp_path, never into the project."""
    import numpy as np
    import rasterio
    from insar_pipeline import client_report
    from insar_pipeline.config import load_config
    cfg = load_config(cfg_path)
    rep, ctx, masks, _ = client_report.compute(cfg)
    rep = json.loads(json.dumps(rep))                    # exactly what build() writes
    paths = client_report.write_rasters(cfg, rep, ctx, out_dir=tmp_path / "rasters")
    vr = next(p for p in paths if p.name in ("vertical_movement_in_per_yr.tif", "vertical_movement_mm_per_yr.tif"))
    with rasterio.open(vr) as r:
        v = r.read(1).astype(float) * (25.4 if "_in_" in vr.name else 1.0)
    means = {name: float(np.nanmean(v[m & ctx.good])) for name, m in masks.items() if (m & ctx.good).any()}
    return rep, means


