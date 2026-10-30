"""Fill MintPy's ERA5 cache in parallel, with the exact request pyaps3 makes.

MintPy's tropospheric step (pyaps3.ECMWFdload) downloads one GRIB per
acquisition date, sequentially, from the CDS queue - ~1.2 min each on
22 Sep 2026, i.e. ~4 h for 252 dates. It skips any file already present,
so this script fetches the missing dates N-wide, writes each to a .part
name and renames it into place atomically. It walks the date list in
REVERSE so it meets the sequential downloader in the middle instead of
racing it date by date.

    python scripts/prefetch_era5.py <config.yaml> [workers]
        the pipeline's own prefetch (stack_backend.start_era5_prefetch): the
        dates, hour and tile MintPy WILL use, worked out from the project
        before MintPy has run - the same thing a stack / phase-linked run now
        starts by itself during coregistration;
    python scripts/prefetch_era5.py <mintpy_dir>/ERA5 <ifgramStack.h5> [workers]
        the original form: tile and hour copied from a GRIB already in the
        folder, dates from the interferogram stack.

Run it from the insar env. The request and the file names come from
stack_backend (era5_request / era5_grib_name), one definition for both forms.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline.stack_backend import fetch_era5_grib  # noqa: E402


def parse_name(sample: str):
    """ERA5_N30_N50_W90_W80_20160609_23.grb -> (snwe, hour, prefix)."""
    parts = sample.split("_")

    def val(tok):
        sign = -1 if tok[0] in "SW" else 1
        return int(sign * float(tok[1:]))
    s, n, w, e = val(parts[1]), val(parts[2]), val(parts[3]), val(parts[4])
    hour = parts[6].split(".")[0]
    return (s, n, w, e), hour, "_".join(parts[:5])


def fetch(args):
    date, fname, snwe, hour = args[:4]
    return date, fetch_era5_grib(date, fname, snwe, hour)


def from_config(path: str, workers: int) -> int:
    from insar_pipeline.config import load_config
    from insar_pipeline.stack_backend import start_era5_prefetch
    st = start_era5_prefetch(load_config(path), workers=workers)
    print(f"{st['status']}: {st['reason']}", flush=True)
    return 0 if st["status"] in ("done", "skipped") else 1


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if sys.argv[1].endswith((".yaml", ".yml")):
        return from_config(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    import h5py
    era5 = Path(sys.argv[1])
    stack = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    samples = sorted(p.name for p in era5.glob("ERA5_*.grb"))
    if not samples:
        print(f"no ERA5_*.grb in {era5} to copy the tile and hour from; use the <config.yaml> form")
        return 2
    snwe, hour, prefix = parse_name(samples[0])
    with h5py.File(stack, "r") as f:
        dates = sorted({d.decode() for pair in f["date"][:] for d in pair})
    todo = [(d, str(era5 / f"{prefix}_{d}_{hour}.grb"), snwe, hour)
            for d in reversed(dates) if not (era5 / f"{prefix}_{d}_{hour}.grb").exists()]
    print(f"{len(dates)} dates, {len(todo)} missing; {workers} workers; area {snwe} hour {hour}",
          flush=True)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (d, status) in enumerate(pool.map(fetch, todo), 1):
            print(f"[{i}/{len(todo)}] {d} {status}", flush=True)
            ok += status in ("ok", "cached", "cached-meanwhile")
            fail += status.startswith("FAILED")
    print(f"done: {ok} ok, {fail} failed", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
