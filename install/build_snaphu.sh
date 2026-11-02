#!/usr/bin/env bash
# Build snaphu 2.0.7 (Stanford) into a conda env's bin/. The pipeline's
# unwrapping uses this standalone binary: ISCE2's embedded snaphu 1.4.2 caps
# connected components at 20 and truncates the minimum component size to 0.
#
#   install/build_snaphu.sh [env]        (default: miaplpy)
#
# Built WITHOUT the CS2 minimum-cost-flow module (-DNO_CS2, and snaphu_cs2.c is
# not compiled). snaphu's README grants the CS2-derived code for "strictly
# noncommercial purposes" only; the rest of snaphu may be used for any purpose
# (no fee may be charged for redistributing it). The pipeline never uses CS2:
# unwrap_crop.py writes INITMETHOD MST, and doctor's known-answer test uses MST.
# A config asking for INITMETHOD MCF now stops with "program not compiled with
# cs2 MCF solver module", which this script checks after building.
#
# Options (environment variables), for offline or test builds:
#   SNAPHU_TARBALL=/path/snaphu-v2.0.7.tar.gz   use a local copy (checksum still verified)
#   SNAPHU_DEST=/some/dir                       install there instead of the env's bin/
set -euo pipefail
ENV_NAME="${1:-miaplpy}"
URL="https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu-v2.0.7.tar.gz"
SHA256="c03ac126f9a964321bb5d6fb5b4004728368da268d7cb8407bb295a8abe5b262"
set +u
CONDA_BASE="$(conda info --base 2>/dev/null || true)"; CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$ENV_NAME"
set -u
DEST="${SNAPHU_DEST:-$CONDA_PREFIX/bin}"
WORK="$(mktemp -d)"; trap 'rm -rf -- "${WORK:?}"' EXIT
if [[ -n "${SNAPHU_TARBALL:-}" ]]; then
    cp -- "$SNAPHU_TARBALL" "$WORK/snaphu.tar.gz"
else
    curl -fsSL "$URL" -o "$WORK/snaphu.tar.gz"
fi
echo "$SHA256  $WORK/snaphu.tar.gz" | sha256sum -c --quiet
tar -xzf "$WORK/snaphu.tar.gz" -C "$WORK"
# One compiler call, the Makefile's sources and -O3 (its -arch flag is macOS-only,
# and a fresh Ubuntu has no make). The env's own compiler: conda's "compilers"
# package sets $CC on activation, so no system build tools are needed.
CCX="${CC:-$(command -v cc || command -v gcc || true)}"
[[ -n "$CCX" ]] || { echo "no C compiler: conda install -n $ENV_NAME -c conda-forge compilers"; exit 1; }
SRC="$WORK/snaphu-v2.0.7/src"
# snaphu 2.0.7 (2020) assigns function pointers across prototypes the way C99
# compilers accepted; GCC 14 made that an error. Compile it as the C it is.
( cd "$SRC" && "$CCX" -O3 -std=gnu99 -DNO_CS2 -Wno-error=incompatible-pointer-types -Wno-error=int-conversion \
      -Wno-error=implicit-function-declaration -w \
      snaphu.c snaphu_tile.c snaphu_solver.c snaphu_io.c snaphu_util.c snaphu_cost.c -o snaphu -lm )

# Check the build before it replaces anything: right version, no CS2 code, and
# an MCF request refused.
BIN="$SRC/snaphu"
OUT="$("$BIN" 2>&1 || true)"                       # snaphu exits 1 when run without arguments
[[ "$OUT" == *"v2.0.7"* ]] || { echo "built snaphu does not report v2.0.7"; exit 1; }
if grep -aq "Initializing flows with MCF algorithm" "$BIN"; then
    echo "built snaphu still contains the CS2 MCF module"; exit 1
fi
head -c $((16 * 16 * 8)) /dev/zero > "$WORK/zero.c8"
printf 'OUTFILE %s\nINFILEFORMAT COMPLEX_DATA\nOUTFILEFORMAT FLOAT_DATA\nINITMETHOD MCF\n' "$WORK/zero.unw" > "$WORK/mcf.conf"
MCF="$("$BIN" -f "$WORK/mcf.conf" "$WORK/zero.c8" 16 2>&1 || true)"
[[ "$MCF" == *"not compiled with cs2"* ]] || { echo "built snaphu did not refuse INITMETHOD MCF: $MCF"; exit 1; }

mkdir -p "$DEST"
install -m 755 "$BIN" "$DEST/snaphu"
echo "snaphu 2.0.7 (without the CS2 module) installed: $DEST/snaphu"
