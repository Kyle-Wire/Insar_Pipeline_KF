"""Gate 1, step 3: group picks into events (PyOcto) and locate them.

Picks are first shifted by the per-station delays from g1_calibrate.py, so what's left is
travel time through uniform crust. PyOcto can then associate with a constant-velocity model and
a tight tolerance. Each event is then relocated by robust least squares on the same corrected
picks, with picks more than 1 s off dropped once.

    python g1_associate.py OUT_DIR [P_THRESHOLD S_THRESHOLD]
writes OUT_DIR/events_p{P}_s{S}.csv and OUT_DIR/assignments_p{P}_s{S}.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyocto
from scipy.optimize import least_squares

from g1_common import BOX, pick_files, stations

MIN_PICKS, MIN_P = 6, 4


def load(out, p_thr, s_thr):
    vel = json.load(open(out / "velocity.json"))
    sta = stations(out)[["id", "latitude", "longitude", "elevation"]]
    counts, delays = vel["pick_counts"], vel["delays"]
    # A station or phase the calibration month never picked takes the network median delay.
    med = {ph: float(np.median([delays[s][ph] for s in delays if counts[s][ph] > 0])) for ph in "PS"}
    delay = {(s, ph): delays[s][ph] if s in delays and counts[s][ph] > 0 else med[ph]
             for s in sta["id"] for ph in "PS"}
    picks = pd.concat(pd.read_csv(f) for f in pick_files(out))
    picks = picks[np.where(picks["phase"] == "P", picks["prob"] >= p_thr, picks["prob"] >= s_thr)]
    picks = picks[picks["station"].isin(sta["id"])].reset_index(drop=True)
    picks["time_raw"] = picks["time"]
    picks["time"] = picks["time"] - [delay[(s, ph)] for s, ph in zip(picks["station"], picks["phase"])]
    return vel, sta, picks


def locate(ev_picks, sta_xyz, vp, vs, start):
    xyz = np.array([sta_xyz[s] for s in ev_picks["station"]])
    v = np.where(ev_picks["phase"] == "P", vp, vs)
    t = ev_picks["time"].to_numpy()

    def resid(m):
        d = np.sqrt(((xyz - m[:3]) ** 2).sum(axis=1))
        return m[3] + d / v - t

    fit = least_squares(resid, start, loss="soft_l1", f_scale=0.3,
                        bounds=([-np.inf, -np.inf, 0, -np.inf], [np.inf, np.inf, 40, np.inf]))
    return fit.x, resid(fit.x)


def main():
    out = Path(sys.argv[1])
    p_thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    s_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    vel, sta, picks = load(out, p_thr, s_thr)
    model = pyocto.VelocityModel0D(p_velocity=vel["vp"], s_velocity=vel["vs"], tolerance=1.0,
                                   association_cutoff_distance=150)
    assoc = pyocto.OctoAssociator.from_area(
        lat=(BOX["minlatitude"] - 0.3, BOX["maxlatitude"] + 0.3),
        lon=(BOX["minlongitude"] - 0.3, BOX["maxlongitude"] + 0.3),
        zlim=(0, 40), time_before=60, velocity_model=model,
        n_picks=MIN_PICKS, n_p_picks=MIN_P, n_s_picks=0, n_p_and_s_picks=0)
    stations = assoc.transform_stations(sta.copy())
    events, assign = assoc.associate(picks, stations)
    print(f"{len(picks)} picks (P>={p_thr}, S>={s_thr}) -> {len(events)} associated events")
    if events.empty:
        return

    sta_xyz = {r.id: np.array([r.x, r.y, r.z]) for r in stations.itertuples()}
    rows, kept = [], []
    for ev in events.itertuples():
        ep = assign[assign["event_idx"] == ev.idx].merge(
            picks[["time_raw"]], left_on="pick_idx", right_index=True)
        m, r = locate(ep, sta_xyz, vel["vp"], vel["vs"], np.array([ev.x, ev.y, ev.z, ev.time]))
        good = np.abs(r) <= 1.0
        if good.sum() < len(r) and good.sum() >= MIN_PICKS:
            ep = ep[good]
            m, r = locate(ep, sta_xyz, vel["vp"], vel["vs"], m)
        n_p, n_s = int((ep["phase"] == "P").sum()), int((ep["phase"] == "S").sum())
        if n_p < MIN_P:
            continue
        rows.append(dict(event=int(ev.idx), time=m[3], x=m[0], y=m[1], z=m[2],
                         n_p=n_p, n_s=n_s, n_sta=ep["station"].nunique(),
                         rms=float(np.sqrt(np.mean(r ** 2)))))
        kept.append(ep.assign(event=int(ev.idx)))
    # PyOcto works in a local km frame; its own transform takes x, y, z back to lat/lon/depth.
    res = assoc.transform_events(pd.DataFrame(rows))
    tag = f"p{p_thr:g}_s{s_thr:g}"
    res.to_csv(out / f"events_{tag}.csv", index=False)
    pd.concat(kept).to_csv(out / f"assignments_{tag}.csv", index=False)
    print(f"{len(res)} located events, median rms {res['rms'].median():.2f} s, "
          f"median stations {res['n_sta'].median():.0f}")


if __name__ == "__main__":
    main()
