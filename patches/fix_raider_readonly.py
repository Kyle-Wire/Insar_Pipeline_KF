#!/usr/bin/env python
"""
Make RAiDER 0.6.0 work with pandas >= 3.0 (copy-on-write).

pandas 3.0 returns a READ-ONLY array from DataFrame.values. RAiDER predates
that and writes into such arrays in place, so every weather-model
preparation dies with "assignment destination is read-only". RAiDER catches
it per-timestep, logs only the message, reports an empty file list, and
finally raises NoWeatherModelData - so the pipeline sees "Weather model
processing failed for all times" and the real one-line cause never surfaces.

Verified:  pandas 3.0.6  ->  pd.DataFrame(...).values.flags.writeable is False

Two sites:

  interpolator.py  fillna3D()   THE ACTUAL FAILURE. `out` comes straight from
                                dfd.values and is written in place. Replaced
                                with np.where, which is equivalent and does
                                not mutate.

  models/hrrr.py   load_weather_hrrr()   PRECAUTIONARY. `lons` comes from
                                xarray .values and is written in place two
                                lines later. It is not what fails today, but
                                it is the same bug one layer down and would
                                surface as soon as the first one is fixed.

Idempotent. Re-run after any rebuild of raider-env - a conda update to
RAiDER reverts both.

    python patches/fix_raider_readonly.py [--env-prefix PATH]
"""
import argparse, pathlib, sys

EDITS = [
    ("RAiDER/interpolator.py",
     "    outmat = np.moveaxis(out, -1, axis)\n"
     "    outmat[np.isnan(outmat)] = fill_value\n",
     "    outmat = np.moveaxis(out, -1, axis)\n"
     "    # np.where, not in-place: `out` derives from pandas .values, which is\n"
     "    # read-only under copy-on-write (pandas >= 3.0).\n"
     "    outmat = np.where(np.isnan(outmat), fill_value, outmat)\n"),
    ("RAiDER/models/hrrr.py",
     "    lons = ds['longitude'].values\n",
     "    # .copy(): xarray .values can be a read-only view, and the\n"
     "    # `lons[lons > 180] -= 360` below writes into it in place.\n"
     "    lons = ds['longitude'].values.copy()\n"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--env-prefix", default=str(pathlib.Path(sys.prefix).parent / "raider-env"))
a = ap.parse_args()
root = pathlib.Path(a.env_prefix) / "lib/python3.11/site-packages"

rc = 0
for rel, old, new in EDITS:
    f = root / rel
    if not f.exists():
        print(f"  MISSING  {f}"); rc = 1; continue
    s = f.read_text()
    # Compare on the code line only, so a reworded comment from an earlier
    # revision of this script still counts as applied.
    code = new.strip().splitlines()[-1].strip()
    if new in s or code in s:
        print(f"  already  {rel}"); continue
    if s.count(old) != 1:
        print(f"  SKIP     {rel}: expected 1 match, found {s.count(old)} "
              f"- RAiDER changed, re-check upstream"); rc = 1; continue
    f.write_text(s.replace(old, new, 1))
    print(f"  patched  {rel}")
sys.exit(rc)
