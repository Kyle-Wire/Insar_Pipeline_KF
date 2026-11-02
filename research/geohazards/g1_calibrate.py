"""Gate 1, step 2: fit a simple velocity model from the catalog's own analyst picks.

No layered New Madrid model turned up in a search, so fit one: straight rays through uniform
crust at speeds vp and vs, plus a fixed delay per station and phase. The delays soak up the
slow embayment sediments under each station. The fit uses the reviewed USGS/NM events and their
analyst picks from a *different* month (the calibration month), so the test month's locations
aren't scored against a model tuned on them.

    python g1_calibrate.py 2026-08-01 2026-09-01 OUT_DIR
writes OUT_DIR/velocity.json  {"vp":..,"vs":..,"delays":{"NM.STA":{"P":..,"S":..}}, ...}
"""
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read_events
from obspy.clients.fdsn import Client
from scipy.optimize import least_squares

from g1_common import BOX, DEG_KM, stations

USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def phase_quakeml(event_id):
    with urllib.request.urlopen(f"{USGS}?eventid={event_id}&format=geojson", timeout=30) as r:
        det = json.load(r)
    url = det["properties"]["products"]["phase-data"][0]["contents"]["quakeml.xml"]["url"]
    with urllib.request.urlopen(url, timeout=30) as r:
        return read_events(io.BytesIO(r.read()))[0]


def main():
    start, end, out = UTCDateTime(sys.argv[1]), UTCDateTime(sys.argv[2]), Path(sys.argv[3])
    sta = stations(out).set_index("id").to_dict("index")
    cat = Client("USGS").get_events(starttime=start, endtime=end, **BOX)
    obs = []  # (event index, station, phase, travel time)
    origins = []
    for ev in cat:
        eid = str(ev.resource_id).split("eventid=")[-1].split("&")[0]
        try:
            qe = phase_quakeml(eid)
        except Exception as e:
            print("skip", eid, type(e).__name__)
            continue
        o = qe.preferred_origin() or qe.origins[0]
        picks = {str(p.resource_id): p for p in qe.picks}
        k = len(origins)
        origins.append((o.latitude, o.longitude, o.depth / 1000))
        for a in o.arrivals:
            p = picks.get(str(a.pick_id))
            if p is None or a.phase[:1] not in "PS":
                continue
            sid = f"{p.waveform_id.network_code}.{p.waveform_id.station_code}"
            if sid in sta:
                obs.append((k, sid, a.phase[:1], p.time - o.time))
    print(f"{len(origins)} calibration events, {len(obs)} analyst picks on our stations")

    sids = sorted({o[1] for o in obs})
    keys = [(s, ph) for s in sids for ph in "PS"]
    kidx = {k: i for i, k in enumerate(keys)}
    ev = np.array([origins[o[0]] for o in obs])
    st = np.array([[sta[o[1]]["latitude"], sta[o[1]]["longitude"], -sta[o[1]]["elevation"] / 1000]
                   for o in obs])
    dx = (st[:, 1] - ev[:, 1]) * DEG_KM * np.cos(np.radians(ev[:, 0]))
    dy = (st[:, 0] - ev[:, 0]) * DEG_KM
    dist = np.sqrt(dx ** 2 + dy ** 2 + (st[:, 2] - ev[:, 2]) ** 2)
    is_s = np.array([o[2] == "S" for o in obs])
    tt = np.array([o[3] for o in obs])
    ki = np.array([kidx[(o[1], o[2])] for o in obs])

    def resid(x):
        v = np.where(is_s, x[1], x[0])
        d = x[2:][ki]
        # Delays are tied weakly to zero so stations with few picks stay sensible.
        return np.concatenate([dist / v + d - tt, 0.1 * x[2:]])

    x0 = np.r_[6.0, 3.5, np.zeros(len(keys))]
    fit = least_squares(resid, x0, loss="soft_l1", f_scale=0.3)
    r = resid(fit.x)[: len(tt)]
    vp, vs = fit.x[:2]
    delays = {s: {ph: round(float(fit.x[2 + kidx[(s, ph)]]), 3) for ph in "PS"} for s in sids}
    counts = {s: {ph: int(np.sum((ki == kidx[(s, ph)]))) for ph in "PS"} for s in sids}
    print(f"vp {vp:.2f} km/s, vs {vs:.2f} km/s, Vp/Vs {vp / vs:.2f}")
    print(f"travel-time residuals: P median |r| {np.median(np.abs(r[~is_s])):.2f} s, "
          f"S median |r| {np.median(np.abs(r[is_s])):.2f} s")
    json.dump(dict(vp=float(vp), vs=float(vs), delays=delays, pick_counts=counts,
                   calibration=dict(start=str(start), end=str(end), events=len(origins),
                                    picks=len(obs))),
              open(out / "velocity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
