"""Shared settings for the Gate 1 New Madrid scripts."""
from pathlib import Path

import pandas as pd
from obspy import Stream, read

BOX = dict(minlatitude=35.7, maxlatitude=37.0, minlongitude=-90.3, maxlongitude=-88.9)
DEG_KM = 111.19
SETS = ["", "_extra"]  # file suffixes of the station sets written by g1_pick.py


def stations(out: Path) -> pd.DataFrame:
    """Every station set downloaded so far, as one table."""
    return pd.concat(pd.read_csv(out / f"stations{s}.csv") for s in SETS
                     if (out / f"stations{s}.csv").exists()).drop_duplicates("id")


def pick_files(out: Path) -> list[Path]:
    return sorted(f for s in SETS for f in (out / f"picks{s}").glob("*.csv"))


def day_stream(out: Path, day: str, headonly: bool = False) -> Stream:
    """One day's waveforms from every station set that has it."""
    st = Stream()
    for s in SETS:
        f = out / f"mseed{s}" / f"{day}.mseed"
        if f.exists():
            st += read(str(f), headonly=headonly)
    return st
