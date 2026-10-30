"""Gate 1 diagnostic: which pretrained picker recovers the analysts' picks best on this network?

For every catalog event on a day we have miniSEED for, cut 2 minutes around the origin, run each
candidate model, and score it against the event's reviewed analyst picks on our stations: the
fraction found within 0.5 s at each threshold, and the timing error of the ones found.

    python g1_pickers.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seisbench.models as sbm
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from g1_calibrate import phase_quakeml
from g1_common import BOX, day_stream, stations

MODELS = [("PhaseNet", "instance"), ("PhaseNet", "stead"), ("PhaseNet", "original"),
          ("EQTransformer", "original"), ("EQTransformer", "instance")]
THRESHOLDS = [0.1, 0.2, 0.3, 0.5]


def main():
    out = Path(sys.argv[1])
    days = sorted(f.stem for f in (out / "mseed").glob("*.mseed"))
    days = [d for d in days if (out / "picks" / f"{d}.csv").exists()]  # day file fully written
    start, end = UTCDateTime(days[0]), UTCDateTime(days[-1]) + 86400
    cat = Client("USGS").get_events(starttime=start, endtime=end, **BOX)
    truth, windows = [], {}
    for ev in cat:
        eid = str(ev.resource_id).split("eventid=")[-1].split("&")[0]
        qe = phase_quakeml(eid)
        o = qe.preferred_origin() or qe.origins[0]
        picks = {str(p.resource_id): p for p in qe.picks}
        for a in o.arrivals:
            p = picks.get(str(a.pick_id))
            if p is not None and a.phase[:1] in "PS":
                w = p.waveform_id
                truth.append(dict(event=eid, station=f"{w.network_code}.{w.station_code}",
                                  channel=w.channel_code, phase=a.phase[:1], time=p.time.timestamp))
        windows[eid] = o.time
    truth = pd.DataFrame(truth)
    print(f"{len(windows)} catalog events, {len(truth)} analyst picks; channels used:")
    print(truth.assign(band=truth.channel.str[:2]).groupby(["band", "phase"]).size().to_string())

    day_streams = {}
    for eid, t0 in windows.items():
        d = t0.strftime("%Y%m%d")
        if d not in day_streams:
            day_streams = {d: day_stream(out, d)}  # one day in memory
        windows[eid] = day_streams[d].slice(t0 - 30, t0 + 90).copy().merge(fill_value=0)
    ours = set(stations(out)["id"])
    truth = truth[truth.station.isin(ours) & truth.channel.str[:2].isin(["HH", "EH"])]
    print(f"{len(truth)} analyst picks on our HH/EH stations")

    rows = []
    for cls, weights in MODELS:
        model = getattr(sbm, cls).from_pretrained(weights).to("cuda")
        got = []
        for eid, st in windows.items():
            for p in model.classify(st, P_threshold=min(THRESHOLDS), S_threshold=min(THRESHOLDS)).picks:
                got.append(dict(event=eid, station=".".join(p.trace_id.split(".")[:2]),
                                phase=p.phase, time=p.peak_time.timestamp, prob=p.peak_value))
        got = pd.DataFrame(got)
        m = truth.merge(got, on=["event", "station", "phase"], how="left", suffixes=("", "_m"))
        m["err"] = (m["time_m"] - m["time"]).abs()
        m = m[m["err"].isna() | (m["err"] <= 0.5)]
        best = m.sort_values("err").drop_duplicates(["event", "station", "phase"])
        best = truth.merge(best, on=["event", "station", "phase", "time", "channel"], how="left")
        for ph in "PS":
            b = best[best.phase == ph]
            row = {"model": f"{cls}/{weights}", "phase": ph, "n": len(b)}
            for t in THRESHOLDS:
                row[f"found@{t}"] = round(float((b["prob"] >= t).mean()), 2)
            row["median_err_s"] = round(float(b.loc[b["prob"] >= 0.3, "err"].median()), 3)
            rows.append(row)
        print(pd.DataFrame(rows[-2:]).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(out / "picker_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
