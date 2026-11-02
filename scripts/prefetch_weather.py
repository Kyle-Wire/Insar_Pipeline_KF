"""Fetch a project's RAiDER weather delays (weather_model, HRRR by default)
ahead of its MintPy run, into <project>/tropo_corrections/<MODEL>/<date>/.

The run (time_series -> weather_raider.build_delay_file) reuses every date
cached here, so a project whose MintPy step is next only writes
inputs/<MODEL>.h5 from them. Dates: the finished time series' when it
exists (mintpy/timeseries.h5), otherwise the planned or downloaded ones.

    python scripts/prefetch_weather.py <config.yaml> [workers]

Run it from the insar env; RAiDER itself runs in the config's raider_env.
Progress is in the project's progress feed (stage weather_raider).
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from insar_pipeline import stack_backend as sb, weather_raider as wr
    from insar_pipeline.config import load_config
    cfg = load_config(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    ts = cfg.mintpy_dir / "timeseries.h5"
    if ts.exists():
        dates, src = wr._read_dates(ts), "mintpy/timeseries.h5"
    else:
        dates, src = sorted(sb.planned_dates(cfg) or sb.available_input_dates(cfg)), "the planned / downloaded dates"
    if not dates:
        print("no dates known yet (no time series, plan or downloads)")
        return 2
    print(f"{len(dates)} dates from {src}; model {wr.model_name(cfg)}; RAiDER env {cfg.raider_env or '(this one)'}",
          flush=True)
    got = wr.fetch_cubes(cfg, dates, workers=workers)
    ok = sum(p is not None for p in got.values())
    print(f"done: {ok} of {len(dates)} dates have a delay cube", flush=True)
    return 0 if ok == len(dates) else 1


if __name__ == "__main__":
    sys.exit(main())
