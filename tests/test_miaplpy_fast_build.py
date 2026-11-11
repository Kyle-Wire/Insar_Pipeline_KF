"""The '.fast' MiaplPy build (10 Nov 2026, patches/miaplpy/speed.patch,
review_2026-11-10/PLAN.md section 8): the covariance goes through BLAS (the
estimate is unchanged to float32 rounding) and all patches share one worker
pool. A series linked with the older build must resume under the new one
and the reverse - or every project would be told to re-link (~12 h) after the
rebuild - while a genuinely different algorithm is still refused."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import miaplpy_backend as mb          # noqa: E402

import test_fix_oct16_security as sec                     # noqa: E402  (the synthetic linked project)

OLD, NEW = "0.2.1+overlapdatum", "0.2.1+overlapdatum.fast"


def test_algorithm_label_strips_speed_only_suffixes():
    assert mb.miaplpy_algorithm(NEW) == OLD
    assert mb.miaplpy_algorithm(OLD) == OLD
    assert mb.miaplpy_algorithm(" " + NEW + " ") == OLD
    assert mb.miaplpy_algorithm(None) == ""
    assert mb.miaplpy_algorithm("0.2.1") == "0.2.1"                     # upstream stays upstream


@pytest.mark.parametrize("linked, installed", [(OLD, NEW), (NEW, OLD), (NEW, NEW)])
def test_speed_only_rebuild_resumes(tmp_path, linked, installed):
    cfg, fp = sec._linked(tmp_path, fp_extra={"miaplpy_version": linked})
    sec._patch_check(cfg, installed)
    assert mb.check_linking_fingerprint(cfg)["sha"] == fp["sha"]


@pytest.mark.parametrize("installed", ["0.2.1", "0.2.1+fast", "0.2.1+overlapdatum.blas", "0.2.2+overlapdatum.blas"])
def test_other_algorithm_still_refused(tmp_path, installed):
    cfg, _ = sec._linked(tmp_path, fp_extra={"miaplpy_version": OLD})
    sec._patch_check(cfg, installed)
    with pytest.raises(RuntimeError, match="MiaplPy build: linked with"):
        mb.check_linking_fingerprint(cfg)


def test_build_script_and_release_carry_the_speed_patch():
    root = Path(mb.__file__).parent
    sh = (root / "patches" / "miaplpy" / "build_miaplpy.sh").read_text()
    assert f'VERSION_LABEL="{NEW}"' in sh
    # the second patch goes on after the datum patch, and the lock moves before the install
    assert sh.index('git apply "$HERE/overlap_window_datum.patch"') < sh.index('git apply "$HERE/speed.patch"')
    assert sh.index("pip config --site set global.constraint") < sh.index("pip install --no-deps")
    assert (root / "patches" / "miaplpy" / "speed.patch").exists()
    assert "patches/miaplpy/speed.patch" in (root / "install" / "make_release.sh").read_text()


def test_speed_patch_changes_speed_only():
    """What the patch touches: the covariance estimator, the job split, the
    pool, the version tag - nothing in the estimators' maths besides the
    covariance product."""
    root = Path(mb.__file__).parent
    patch = (root / "patches" / "miaplpy" / "speed.patch").read_text()
    files = sorted({l.split(" b/")[-1] for l in patch.splitlines() if l.startswith("diff --git")})
    assert files == ["src/miaplpy/lib/utils.pyx", "src/miaplpy/miaplpyApp.py", "src/miaplpy/phase_linking.py",
                     "src/miaplpy/version.py"]
    assert "+        number_of_nodes = 1" in patch
    assert "imap_unordered(func, box_list, chunksize=1)" in patch
    assert "a @ a.conj().T" in patch
    assert "Tag('0.2.1+overlapdatum.fast'" in patch
