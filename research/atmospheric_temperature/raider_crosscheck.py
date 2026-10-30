"""Cross-check refractivity.py against RAiDER's own code on one column.

RAiDER cannot share an environment with ISCE2 (tropo_disp.delay_cube_for_date),
so this file is two things:
  * run under raider-env's python as a script, it reads a column
    {z (m), t (K), p (hPa), e (hPa)} as JSON on stdin, puts it into a RAiDER
    HRRR WeatherModel object and runs RAiDER's _get_wet_refractivity,
    _get_hydro_refractivity and _getZTD (the code that made the cached cubes),
    and prints {"hydro": [...], "wet": [...], "raider": version, "k": [...]}
    per level, metres;
  * imported from the insar env, run(col) does that through a subprocess.
Nothing is downloaded: the HRRR object is only a container for RAiDER's
refractivity constants and integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def default_python() -> Path:
    from insar_pipeline import settings
    return settings.conda_base() / "envs" / "raider-env" / "bin" / "python"


def run(col: dict, python: str | Path | None = None, timeout: int = 120) -> dict:
    py = Path(python) if python else default_python()
    payload = json.dumps({k: [float(v) for v in col[k]] for k in ("z", "t", "p", "e")})
    r = subprocess.run([str(py), "-I", str(Path(__file__).resolve())], input=payload, text=True,
                       capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"RAiDER cross-check failed ({r.returncode}): {(r.stderr or r.stdout)[-600:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> int:
    import numpy as np
    import RAiDER
    from RAiDER.models.hrrr import HRRR
    col = json.loads(sys.stdin.read())
    m = HRRR()
    m._zs = np.asarray(col["z"], dtype="float64")
    shape = (1, 1, -1)
    m._t = np.asarray(col["t"], dtype="float64").reshape(shape)
    m._p = (np.asarray(col["p"], dtype="float64") * 100.0).reshape(shape)   # RAiDER works in Pa
    m._e = (np.asarray(col["e"], dtype="float64") * 100.0).reshape(shape)
    m._get_wet_refractivity()
    m._get_hydro_refractivity()
    m._getZTD()
    print(json.dumps({"hydro": m._hydrostatic_ztd[0, 0].tolist(), "wet": m._wet_ztd[0, 0].tolist(),
                      "raider": getattr(RAiDER, "__version__", "unknown"),
                      "k_per_Pa": [float(m._k1), float(m._k2), float(m._k3)]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
