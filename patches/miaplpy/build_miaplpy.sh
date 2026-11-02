#!/usr/bin/env bash
# Rebuild the PATCHED MiaplPy into a conda env, from upstream + our patch, and
# prove it with the functional check. This is the only supported way to
# (re)install MiaplPy for this pipeline.
#
#   insar_pipeline/patches/miaplpy/build_miaplpy.sh            # env "miaplpy"
#   insar_pipeline/patches/miaplpy/build_miaplpy.sh other_env
#   MIAPLPY_SRC=/home/kylew/src/MiaplPy build_miaplpy.sh      # build from a local clone (offline)
#
# What it does: clones upstream MiaplPy at the pinned commit (or copies
# MIAPLPY_SRC), applies overlap_window_datum.patch, installs it into the env
# with no dependency changes, writes the env's pip constraint that stops pip
# replacing it, and runs check_miaplpy_patch.py. Exit non-zero if any step or
# the check fails. See README.md here for what the patch is and why.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-miaplpy}"
UPSTREAM_URL="https://github.com/insarlab/MiaplPy.git"
UPSTREAM_COMMIT="0d5f4b0cf0de46658c31b850a25c32a5258b773d"
VERSION_LABEL="0.2.1+overlapdatum"

set +u                                  # conda's activate hooks (ISCE's) read unset variables
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}")"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
PREFIX="$CONDA_PREFIX"

WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK:?}"' EXIT
if [[ -n "${MIAPLPY_SRC:-}" ]]; then
    git clone -q "$MIAPLPY_SRC" "$WORK/MiaplPy"
else
    git clone -q "$UPSTREAM_URL" "$WORK/MiaplPy"
fi
cd "$WORK/MiaplPy"
git checkout -q "$UPSTREAM_COMMIT"
git apply --check "$HERE/overlap_window_datum.patch"
git apply "$HERE/overlap_window_datum.patch"
grep -q "$VERSION_LABEL" src/miaplpy/version.py

# (a constraint already in place pins exactly this build, so it allows this install)
pip install --no-deps --no-build-isolation --force-reinstall --quiet .

# pin it: any later pip install/upgrade of miaplpy in this env must match this build
echo "miaplpy==${VERSION_LABEL}" > "$HERE/constraints.txt"
pip config --site set global.constraint "$HERE/constraints.txt" >/dev/null

cd /
python "$HERE/check_miaplpy_patch.py"
echo "Patched MiaplPy ${VERSION_LABEL} installed into ${ENV_NAME} (${PREFIX}) and verified."
