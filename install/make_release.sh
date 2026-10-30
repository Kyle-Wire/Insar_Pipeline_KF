#!/usr/bin/env bash
# Package the pipeline for a coworker inside the operating organisation: one .tar.gz built from a
# WHITELIST of what the software needs (code, scripts, installer, MiaplPy patch,
# tests, the Indiana example mine layers, docs).
# The tarball is INTERNAL to the operating organisation: the machine's local-only tests
# (tests/test_local_*.py) carry client project details, so RELEASE.txt says so
# on its own line (decision D14, 7 Oct 2026). Project configs are not in the
# package: they live in the projects folder's configs/ (28 Oct 2026, Kyle: the
# example and experiment configs left the repository).
# Everything else stays behind: projects, backups, review notes, caches,
# personal settings (.claude/), Windows Zone.Identifier files, stray root files
# (config.yaml, project_zone.geojson) and anyone's credentials.
#
#   install/make_release.sh [output_dir]      (default: ~)
#
# Output: <output_dir>/insar_pipeline_<version>_<code hash>.tar.gz with one
# top folder, insar_pipeline/, and insar_pipeline/RELEASE.txt naming the build;
# beside it <same name>.tar.gz.sha256 (sha256sum -c checks a copy).
# The version is pyproject.toml's (bump it for each release); the code hash is
# version.code_hash(), which changes with any edit to the package's .py files.
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$HOME}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
VER="$(cd "$(dirname "$PKG")" && python3 -c 'from insar_pipeline import version; print(version.__version__)' 2>/dev/null || true)"
HASH="$(cd "$(dirname "$PKG")" && python3 -c 'from insar_pipeline import version; print(version.code_hash())' 2>/dev/null || true)"
VER="${VER:-$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$PKG/pyproject.toml" | head -1)}"
NAME="insar_pipeline_${VER:-$(date +%Y%m%d)}${HASH:+_$HASH}"
# Printed in RELEASE.txt and at the end of the build (install/README.md says the same).
INTERNAL="internal: contains client project configurations; do not distribute outside the operating organisation"

# What ships, as paths relative to the package folder (bash globs; a pattern
# that matches nothing is an error, so a renamed file cannot drop out silently).
WHITELIST=(
    "*.py"                                  # the package's modules
    # the two browser pages the code serves (map_builder.PAGE, results_viewer.PAGE);
    # results_viewer.html was missing until 16 Oct 2026, so 'm' failed on an install
    project_builder.html results_viewer.html pyproject.toml requirements.txt cdsapirc.template
    # the results viewer's scripts, styles and vendored libraries (results_viewer.ASSETS,
    # 19 Oct 2026: no CDN) and their node tests
    "viewer_assets/*.js" "viewer_assets/*.css" "viewer_assets/tests/*.js"
    "viewer_assets/vendor/*.js" "viewer_assets/vendor/*.css" viewer_assets/vendor/LICENSES.txt
    "viewer_assets/vendor/images/*.png" viewer_assets/vendor/images/spritesheet.svg
    # the app's page, script, style and icon (app_server.ASSETS; it borrows Leaflet from viewer_assets)
    app_assets/app.html app_assets/app.js app_assets/app.css app_assets/icon.png
    "app_assets/fonts/*.woff2" app_assets/fonts/LICENSES.txt
    # the Windows setup and the desktop icon's launcher (install/windows/setup.ps1)
    install/windows/setup.ps1 "install/windows/Setup InSAR Pipeline.cmd" "install/windows/InSAR Pipeline.vbs"
    install/windows/insar.ico
    README.md USER_GUIDE.md INSAR_RUNBOOK.md THIRD_PARTY.md VALIDATION.md CHANGELOG.md CONTRIBUTING.md LICENSE
    # (HANDOFF.md, the engineering log, holds client project history: it stays on the
    # developer's machine and is not in the repository - 16 Oct 2026)
    "scripts/*.py" scripts/insar scripts/map_builder
    install/README.md "install/*.sh" "install/envs/*.yml"
    patches/fix_raider_readonly.py
    patches/miaplpy/README.md patches/miaplpy/build_miaplpy.sh
    patches/miaplpy/check_miaplpy_patch.py patches/miaplpy/overlap_window_datum.patch
    "tests/*.py"
    # the retired topsApp runbook, kept for the record (INSAR_RUNBOOK.md links it);
    # the archived configs beside it stay out
    configs/archive/INSAR_RUNBOOK_topsApp_2026.md
    # public Indiana mine layers (IGWS, Indiana DNR CMIS); see context/SOURCE.txt
    context/SOURCE.txt "context/*.geojson"
)
# Never shipped, whatever the whitelist says (checked on the staged copy).
FORBIDDEN='(^|/)(\.claude|backups|review_[^/]*|__pycache__|\.pytest_cache|[^/]*\.egg-info)(/|$)|:Zone\.Identifier$|\.pyc$|^insar_pipeline/(config\.yaml|project_zone\.geojson)$|(^|/)constraints\.txt$|^insar_pipeline/configs/archive/.*\.ya?ml$'

STAGE="$(mktemp -d)"; trap 'rm -rf -- "${STAGE:?}"' EXIT
DEST="$STAGE/insar_pipeline"
mkdir -p "$DEST"
cd "$PKG"
shopt -s nullglob
for pat in "${WHITELIST[@]}"; do
    if [[ "$pat" == *[*?[]* ]]; then
        # shellcheck disable=SC2206
        hits=( $pat )
    else
        hits=( "$pat" )             # a plain name, spaces and all ("Setup InSAR Pipeline.cmd")
    fi
    found=0
    for f in "${hits[@]}"; do
        [[ -f "$f" ]] || continue
        case "$f" in *:Zone.Identifier) continue;; esac
        mkdir -p "$DEST/$(dirname "$f")"
        cp -p -- "$f" "$DEST/$f"
        found=1
    done
    if (( found == 0 )); then
        echo "make_release: nothing matches '$pat'"; exit 1
    fi
done
shopt -u nullglob

{
    echo "release   $NAME"
    echo "use       $INTERNAL"
    echo "version   ${VER:-unknown}"
    echo "code hash ${HASH:-unknown}"
    echo "built     $(date '+%Y-%m-%d %H:%M %Z') on $(hostname)"
    echo "files     $(cd "$STAGE" && find insar_pipeline -type f | wc -l)"
} > "$DEST/RELEASE.txt"

BAD="$(cd "$STAGE" && find insar_pipeline -print | grep -E "$FORBIDDEN" || true)"
if [[ -n "$BAD" ]]; then
    echo "make_release: refusing, the staged copy holds files that must not ship:"; echo "$BAD"; exit 1
fi

tar -C "$STAGE" --owner=0 --group=0 -czf "$OUT/$NAME.tar.gz" insar_pipeline
# The checksum beside it, in sha256sum's format with the file name only, so the
# coworker checks the copy they received with `sha256sum -c <name>.sha256` in the
# folder holding both (16 Oct 2026, readiness review BLD-05).
(cd "$OUT" && sha256sum -- "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
# The Windows setup beside them, so a coworker can start from Windows: double-click
# "Setup InSAR Pipeline.cmd" in the folder holding all four (it installs Ubuntu,
# unpacks this release into it, runs install.sh and makes the desktop icon).
cp -p -- "$PKG/install/windows/setup.ps1" "$PKG/install/windows/Setup InSAR Pipeline.cmd" "$OUT/"
echo "$OUT/$NAME.tar.gz ($(du -h "$OUT/$NAME.tar.gz" | cut -f1), $(tar -tzf "$OUT/$NAME.tar.gz" | grep -vc '/$') files)"
echo "SHA-256: $(cut -d' ' -f1 "$OUT/$NAME.tar.gz.sha256")  ($OUT/$NAME.tar.gz.sha256)"
echo "This release is ${INTERNAL}."
echo "First install:  on Windows, double-click 'Setup InSAR Pipeline.cmd' beside the release file;"
echo "                in Ubuntu, cd ~ && tar -xzf $NAME.tar.gz && bash insar_pipeline/install/install.sh"
echo "Updating:       see install/README.md, 'Updating' (move the old folder aside first)"
