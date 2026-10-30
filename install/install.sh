#!/usr/bin/env bash
# Install the InSAR pipeline on this computer (Ubuntu, or Ubuntu on Windows via WSL2).
#
#   bash install/install.sh                    # everything; safe to re-run
#   bash install/install.sh --no-credentials   # skip the login prompts
#
# What it does, in order (each step is skipped when already done):
#   1. Miniforge (conda) into ~/miniforge3 if no conda is found
#   2. conda environments "insar", "miaplpy", "burst2safe", "raider-env" (NOAA HRRR
#      weather delays through RAiDER) and "dolphin" (phase linking for the NISAR phase-linked
#      route) from install/envs/*.yml
#      (exact versions, conda-forge only; ~9 GB, 10-30 min). An environment that
#      already exists is kept, but checked against the pins: a mismatch is
#      reported with the command that rebuilds it (nothing is deleted here).
#   3. this pipeline into the insar environment (pip install -e)
#   4. snaphu 2.0.7 without the CS2 module into the miaplpy environment
#      (install/build_snaphu.sh); a copy that still contains CS2 is rebuilt
#   5. the patched MiaplPy (patches/miaplpy/build_miaplpy.sh), and its pip
#      constraint pointed at this folder
#   6. logins: NASA Earthdata (~/.netrc) and Copernicus CDS (~/.cdsapirc)
#   7. the projects folder, and the "insar" command in ~/.local/bin
#   8. insar doctor
#
# Updating: move the old folder aside and unpack the new release in its place
# (install/README.md, "Updating"), then run this script from the new folder.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
ASK_CREDS=1
[[ "${1:-}" == "--no-credentials" ]] && ASK_CREDS=0
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*"; }

# The package is imported as "insar_pipeline": the `insar` command puts the
# folder ABOVE this one on the Python path, so the folder must have that name.
if [[ "$(basename "$PKG")" != "insar_pipeline" ]]; then
    echo "This folder is called '$(basename "$PKG")'; it must be called 'insar_pipeline'."
    echo "Rename it (mv \"$PKG\" \"$(dirname "$PKG")/insar_pipeline\") and run install.sh again."
    exit 1
fi
VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$PKG/pyproject.toml" | head -1)"
echo "InSAR pipeline ${VERSION:-(version unknown)} at $PKG"
[[ -f "$PKG/RELEASE.txt" ]] && sed 's/^/  /' "$PKG/RELEASE.txt"

say "1/8 conda"
if ! command -v conda >/dev/null 2>&1 && [[ ! -x "$HOME/miniforge3/bin/conda" ]]; then
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm -f /tmp/miniforge.sh
fi
CONDA_BASE="$(conda info --base 2>/dev/null || true)"; CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
set +u; source "$CONDA_BASE/etc/profile.d/conda.sh"; set -u
SOLVER=conda; command -v mamba >/dev/null 2>&1 && SOLVER=mamba
echo "conda at $CONDA_BASE ($SOLVER)"

# Does an existing environment hold exactly the pinned conda packages?
# Prints "N of M pinned packages match" and the first differences; exit 1 on any.
pins_match() {
    "$CONDA_BASE/bin/python" - "$1" "$2" <<'PY'
import glob, json, os, re, sys
yml, prefix = sys.argv[1], sys.argv[2]
want = {}
for line in open(yml):
    m = re.match(r"^  - ([A-Za-z0-9_.\-]+)=([^=\s]+)\s*$", line)
    if m:
        want[m.group(1)] = m.group(2)
have = {}
for f in glob.glob(os.path.join(prefix, "conda-meta", "*.json")):
    try:
        d = json.load(open(f))
    except (OSError, ValueError):
        continue
    have[d.get("name")] = d.get("version")
bad = [f"{k}: {have.get(k, 'not installed')} (pinned {v})" for k, v in sorted(want.items()) if have.get(k) != v]
print(f"{len(want) - len(bad)} of {len(want)} pinned packages match")
for b in bad[:12]:
    print("    " + b)
sys.exit(1 if bad or not want else 0)
PY
}

for ENV in insar miaplpy burst2safe raider-env dolphin; do
    say "2/8 environment $ENV"
    YML="$HERE/envs/$ENV.yml"
    if [[ -d "$CONDA_BASE/envs/$ENV" ]]; then
        if pins_match "$YML" "$CONDA_BASE/envs/$ENV"; then
            echo "exists and matches install/envs/$ENV.yml - kept"
        else
            warn "environment $ENV exists but does not match install/envs/$ENV.yml (listed above)."
            warn "Results may differ from the validated build. To rebuild it, run:"
            warn "    conda env remove -n $ENV && bash $HERE/install.sh"
        fi
    else
        $SOLVER env create -y -f "$YML"
    fi
done

say "3/8 the pipeline"
set +u; conda activate insar; set -u
pip install --quiet --no-deps --no-build-isolation -e "$PKG"
python -c "import insar_pipeline, sys; print('insar_pipeline at', insar_pipeline.__file__)"

say "4/8 snaphu 2.0.7 (without the CS2 module)"
SN="$CONDA_BASE/envs/miaplpy/bin/snaphu"
SV="$("$SN" 2>&1 || true)"
if [[ "$SV" == *"v2.0.7"* ]] && ! grep -aq "Initializing flows with MCF algorithm" "$SN"; then
    echo "present"
else
    [[ "$SV" == *"v2.0.7"* ]] && echo "the installed snaphu contains the CS2 module (noncommercial terms); rebuilding without it"
    bash "$HERE/build_snaphu.sh" miaplpy
fi

say "5/8 patched MiaplPy"
if "$CONDA_BASE/envs/miaplpy/bin/python" "$PKG/patches/miaplpy/check_miaplpy_patch.py" >/dev/null 2>&1; then
    echo "present and verified"
    # The pip lock that stops pip replacing the patched build names a file in
    # the folder that built it; point it at this folder, so moving or deleting
    # an old copy of the pipeline never leaves pip a missing constraint file.
    LABEL="$(sed -n 's/^VERSION_LABEL="\(.*\)"/\1/p' "$PKG/patches/miaplpy/build_miaplpy.sh")"
    echo "miaplpy==${LABEL}" > "$PKG/patches/miaplpy/constraints.txt"
    "$CONDA_BASE/envs/miaplpy/bin/python" -m pip config --site set global.constraint \
        "$PKG/patches/miaplpy/constraints.txt" >/dev/null
    echo "pip lock: miaplpy==${LABEL} ($PKG/patches/miaplpy/constraints.txt)"
else
    bash "$PKG/patches/miaplpy/build_miaplpy.sh" miaplpy
fi

say "6/8 logins"
if [[ $ASK_CREDS == 1 ]]; then
    if ! grep -q "urs.earthdata.nasa.gov" "$HOME/.netrc" 2>/dev/null; then
        echo "NASA Earthdata login (free: https://urs.earthdata.nasa.gov/users/new) - downloads Sentinel-1, OPERA, GNSS ionosphere maps."
        read -r -p "  Earthdata username (Enter to skip): " EU
        if [[ -n "$EU" ]]; then
            read -r -s -p "  Earthdata password: " EP; echo
            printf 'machine urs.earthdata.nasa.gov login %s password %s\n' "$EU" "$EP" >> "$HOME/.netrc"
            chmod 600 "$HOME/.netrc"
        fi
    else echo "Earthdata login present"; fi
    if [[ ! -f "$HOME/.cdsapirc" ]]; then
        echo "Copernicus CDS key (free: https://cds.climate.copernicus.eu - profile page, 'API token') - ERA5 weather."
        echo "Also accept the 'CC-BY licence' on the ERA5 pressure-levels dataset page (Download data -> Terms of use)."
        read -r -p "  CDS API token (Enter to skip): " CK
        if [[ -n "$CK" ]]; then
            printf 'url: https://cds.climate.copernicus.eu/api\nkey: %s\n' "$CK" > "$HOME/.cdsapirc"
            chmod 600 "$HOME/.cdsapirc"
        fi
    else echo "CDS key present"; fi
fi

say "7/8 projects folder and the insar command"
# The default is the folder already in use ($INSAR_PROJECTS, else the one saved
# in settings.json), so re-running the installer - with --no-credentials too -
# never moves a projects folder set with `insar doctor --projects` back home.
DEFAULT="$(python -c 'from insar_pipeline import settings; print(settings.projects_root())' 2>/dev/null | tail -n 1 || true)"
DEFAULT="${DEFAULT:-${INSAR_PROJECTS:-$HOME/insar_projects}}"
if [[ $ASK_CREDS == 1 ]]; then read -r -p "  Projects folder [$DEFAULT]: " PR; fi
PR="${PR:-$DEFAULT}"
mkdir -p "$PR"
python - "$PR" <<'PY'
import sys
from insar_pipeline import settings
print("settings:", settings.save(projects_root=sys.argv[1]))
PY
mkdir -p "$HOME/.local/bin"
for CMD in insar; do
    ln -sf "$PKG/scripts/insar" "$HOME/.local/bin/$CMD"
done
echo "insar -> $(readlink -f "$HOME/.local/bin/insar")"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"; echo "added ~/.local/bin to PATH (new terminals)";; esac

say "8/8 check"
python -m insar_pipeline.dashboard doctor || true
echo
echo "Licences and the credits every client output must carry: $PKG/THIRD_PARTY.md"
echo "This software is for use inside the operating organisation. It includes GPL-3 components (MintPy, MiaplPy):"
echo "do not give it to anyone outside the operating organisation without a licence review."
echo
echo "Done. Open a new terminal and type:  insar      (press ? in it for the user guide)"
