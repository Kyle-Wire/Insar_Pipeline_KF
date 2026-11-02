"""One zip for the custodian when something goes wrong: `insar diagnose <project>`
(readiness review ERR-04, 2 Nov 2026).

    insar diagnose <project name or folder> [--out DIR]

Writes <projects root>/diagnostics/<project>_<YYYYmmdd-HHMMSS>.zip (or into
--out) holding what the custodian needs to reproduce a defect, and nothing
that identifies the client's ground beyond what the run records already say:

  - run_result.json, run_result_history.jsonl, run.json, progress.jsonl
  - quality_assessment.json
  - the last three run logs (logs/*.log)
  - the JSON records in deliverables/ (client_report.json, gnss_tie.json,
    provenance.json, summary files), deliverables/config_snapshot.yaml
  - the project's config file
  - `insar --version` and `insar doctor --quick` output
  - MANIFEST.txt: every file in the zip, its size and sha256

Never: context/ (client outlines and uploaded originals), the report, KMZ,
rasters, point files or map exports, credentials, settings or the
notifications log. Text files have the home folder written '~' and the
Earthdata user name masked.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

RECORDS = ("run_result.json", "run_result_history.jsonl", "run.json", "progress.jsonl",
           "quality_assessment.json")
LOGS_KEPT = 3
# deliverables/ files that are records, not client products
DELIVERABLE_RECORDS = ("*.json", "config_snapshot.yaml")
NEVER = ("context", ".netrc", ".cdsapirc", "settings.json")


# Text files are written with the home folder as '~' (provenance.portable) and the
# Earthdata user name masked: the zip leaves the machine, and neither is needed to
# reproduce a defect.
TEXT_SUFFIXES = (".json", ".jsonl", ".log", ".yaml", ".yml", ".txt")
_LOGIN = re.compile(r"(login for ')[^']*(')")


def redact(text: str) -> str:
    from .provenance import portable
    return _LOGIN.sub(r"\1<user>\2", portable(text))


def _doctor_quick(timeout: int = 300) -> str:
    try:
        r = subprocess.run([sys.executable, "-m", "insar_pipeline.doctor", "--quick"], capture_output=True,
                           text=True, timeout=timeout, cwd=str(Path(__file__).resolve().parents[1]))
        return r.stdout + (("\n" + r.stderr) if r.stderr else "")
    except Exception as exc:                                   # noqa: BLE001 - the bundle is written regardless
        return f"insar doctor --quick did not run: {exc}\n"


def collect(P: Path, cfg_path: str | None = None) -> list[tuple[str, Path]]:
    """[(name in the zip, file)] for one project folder."""
    P = Path(P)
    out = [(n, P / n) for n in RECORDS if (P / n).is_file()]
    logs = sorted((P / "logs").glob("*.log"), key=lambda f: f.stat().st_mtime)[-LOGS_KEPT:] \
        if (P / "logs").is_dir() else []
    out += [(f"logs/{f.name}", f) for f in logs]
    d = P / "deliverables"
    if d.is_dir():
        seen = set()
        for pat in DELIVERABLE_RECORDS:
            for f in sorted(d.glob(pat)):
                if f.is_file() and f.name not in seen:
                    seen.add(f.name)
                    out.append((f"deliverables/{f.name}", f))
    if cfg_path and Path(cfg_path).is_file():
        out.append((f"config/{Path(cfg_path).name}", Path(cfg_path)))
    for name, _f in out:
        assert not any(part in NEVER for part in Path(name).parts), name
    return out


def write(P: Path, cfg_path: str | None, dest_dir: Path, name: str, doctor: bool = True) -> Path:
    from . import version
    files = collect(P, cfg_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zpath = dest_dir / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    lines = [f"insar diagnose for project '{name}', {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
             f"build: {version.label()}", ""]
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for arc, f in files:
            data = f.read_bytes()
            if f.suffix in TEXT_SUFFIXES:
                data = redact(data.decode("utf-8", "replace")).encode()
            z.writestr(arc, data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {len(data):>10}  {arc}")
        extra = {"version.txt": version.label() + "\n"}
        if doctor:
            extra["doctor_quick.txt"] = redact(_doctor_quick())
        for arc, text in extra.items():
            data = text.encode()
            z.writestr(arc, data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {len(data):>10}  {arc}")
        z.writestr("MANIFEST.txt", "\n".join(lines) + "\n")
    return zpath


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: insar diagnose <project name or folder> [--out DIR]")
        return 0 if argv else 2
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1]).expanduser()
        del argv[i:i + 2]
    from . import settings, storage
    try:
        pname, P, cfg_path, _cfg = storage.find(argv[0])
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    dest = out or settings.projects_root() / "diagnostics"
    z = write(P, cfg_path, dest, pname, doctor="--no-doctor" not in argv)
    print(f"Wrote {z} ({z.stat().st_size / 1e6:.1f} MB). Send it to the custodian; it holds no client outlines, "
          "report, maps or point files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
