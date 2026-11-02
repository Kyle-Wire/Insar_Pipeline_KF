"""Is this installed release the one that was built? (readiness review SEC-06)

make_release.sh writes MANIFEST.sha256 into every release: one sha256sum line
per shipped file, paths relative to the package folder. `insar doctor` reads it
here and reports any shipped file that is missing or changed since the build:
the env files, the MiaplPy patch, the installer, the served pages and their
scripts, and the code itself. A git checkout has no manifest; that is not a
fault (the developer's copy is checked by git instead).

    check(package_dir) -> (ok, message, changed_or_missing)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MANIFEST = "MANIFEST.sha256"


def read(manifest: Path) -> dict:
    """{relative path: sha256} from sha256sum lines ('<hex>  <path>')."""
    out = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        if len(digest) == 64 and name:
            out[name.lstrip("*")] = digest.lower()
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(package_dir: Path) -> tuple[bool, str, list[str]]:
    package_dir = Path(package_dir)
    m = package_dir / MANIFEST
    if not m.is_file():
        return True, f"no {MANIFEST} (a development checkout, not an installed release)", []
    try:
        want = read(m)
    except (OSError, UnicodeDecodeError) as exc:
        return False, f"{MANIFEST} unreadable ({exc})", []
    if not want:
        return False, f"{MANIFEST} lists no files", []
    bad = []
    for rel, digest in sorted(want.items()):
        f = package_dir / rel
        if not f.is_file():
            bad.append(f"{rel} (missing)")
        elif _sha256(f) != digest:
            bad.append(f"{rel} (changed)")
    if bad:
        return False, f"{len(bad)} of {len(want)} shipped files differ from the release: " + ", ".join(bad[:5]) \
            + (" ..." if len(bad) > 5 else ""), bad
    return True, f"all {len(want)} shipped files match the release manifest", []
