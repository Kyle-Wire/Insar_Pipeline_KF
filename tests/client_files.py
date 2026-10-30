"""Files a test may need that are not in the repository (16 Oct 2026).

Client project configs and report areas, the golden client numbers, the
review and validation records and the code snapshots stay on the operator's
machine (README.md, "This repository"). A test that reads one runs where it
exists and skips on a fresh clone, saying which file and why."""
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]


def need(path, why: str = "client data, kept out of the repository") -> Path:
    """path (absolute, or relative to the package folder) if it exists;
    otherwise the calling test skips."""
    p = Path(path)
    p = p if p.is_absolute() else PKG / p
    if not p.exists():
        pytest.skip(f"{p.relative_to(PKG) if p.is_relative_to(PKG) else p} is not here ({why})")
    return p


def have(path) -> bool:
    p = Path(path)
    return (p if p.is_absolute() else PKG / p).exists()
