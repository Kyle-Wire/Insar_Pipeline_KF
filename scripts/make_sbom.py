"""A software bill of materials for a release, from the pinned environment files.

    python scripts/make_sbom.py [install/envs] > SBOM.cdx.json

Reads every conda environment file in install/envs/ (the full resolved exports
the installer builds from: conda packages as name=version, pip packages as
name==version) and writes one CycloneDX 1.5 JSON document listing each package
once per environment, with its package URL (pkg:conda/conda-forge/... or
pkg:pypi/...) and the environment it belongs to. make_release.sh puts it in
every release as SBOM.cdx.json (readiness review DEP-01, 2 Nov 2026).

Standard library only: it runs wherever make_release.sh runs, before any
environment exists. It reads the files' simple 'name=version' lines itself
rather than a YAML parser for the same reason.
"""
from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
_CONDA = re.compile(r"^\s*-\s+([A-Za-z0-9_.\-]+)=([^=\s#]+)(?:=([^\s#]+))?\s*(?:#.*)?$")
_PIP = re.compile(r"^\s*-\s+([A-Za-z0-9_.\-\[\]]+)==([^\s#;]+)")


def read_env(path: Path) -> dict:
    """{'name': env name, 'channels': [...], 'conda': [(name, version, build)], 'pip': [(name, version)]}"""
    out = {"name": path.stem, "channels": [], "conda": [], "pip": []}
    section = None
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if re.match(r"^name:\s*", line):
            out["name"] = line.split(":", 1)[1].split("#", 1)[0].strip()
            continue
        if re.match(r"^channels:\s*$", line):
            section = "channels"
            continue
        if re.match(r"^dependencies:\s*$", line):
            section = "deps"
            continue
        if re.match(r"^\s+-\s+pip:\s*$", line):
            section = "pip"
            continue
        if re.match(r"^\S", line):
            section = None
            continue
        if section == "channels":
            out["channels"].append(line.strip().lstrip("- ").strip())
        elif section == "pip":
            m = _PIP.match(line)
            if m:
                out["pip"].append((m.group(1), m.group(2)))
        elif section == "deps":
            m = _CONDA.match(line)
            if m:
                out["conda"].append((m.group(1), m.group(2), m.group(3) or ""))
    return out


def bom(env_dir: Path, version: str = "", code_hash: str = "") -> dict:
    comps = []
    for f in sorted(env_dir.glob("*.yml")):
        env = read_env(f)
        chan = (env["channels"] or ["conda-forge"])[0]
        for name, ver, build in env["conda"]:
            purl = f"pkg:conda/{chan}/{name}@{ver}" + (f"?build={build}" if build else "")
            comps.append({"type": "library", "name": name, "version": ver, "purl": purl,
                          "bom-ref": f"{env['name']}:{purl}",
                          "properties": [{"name": "insar:environment", "value": env["name"]},
                                         {"name": "insar:source", "value": f"conda ({chan})"}]})
        for name, ver in env["pip"]:
            purl = f"pkg:pypi/{name.lower()}@{ver}"
            comps.append({"type": "library", "name": name, "version": ver, "purl": purl,
                          "bom-ref": f"{env['name']}:{purl}",
                          "properties": [{"name": "insar:environment", "value": env["name"]},
                                         {"name": "insar:source", "value": "pip (PyPI)"}]})
    meta = {"timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "component": {"type": "application", "name": "insar_pipeline", "version": version or "unknown"},
            "properties": [{"name": "insar:code_hash", "value": code_hash or "unknown"}]}
    return {"bomFormat": "CycloneDX", "specVersion": "1.5", "serialNumber": f"urn:uuid:{uuid.uuid4()}",
            "version": 1, "metadata": meta, "components": comps}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    env_dir = Path(argv[0]) if argv else PKG / "install" / "envs"
    version = argv[1] if len(argv) > 1 else ""
    code_hash = argv[2] if len(argv) > 2 else ""
    if not any(env_dir.glob("*.yml")):
        print(f"make_sbom: no environment files in {env_dir}", file=sys.stderr)
        return 1
    json.dump(bom(env_dir, version, code_hash), sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
