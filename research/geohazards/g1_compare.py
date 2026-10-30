"""Gate 1, step 4: score our events against the USGS catalog, and plot what doesn't match.

- Interior: catalog epicentres inside the convex hull of our stations (outside it, locations
  are poorly constrained and the pass test doesn't apply).
- Match: our origin within 3 s and 15 km of a catalog origin, one-to-one, closest in time first.
- Every one of our events without a catalog match gets a waveform plot (vertical components,
  nearest stations first, assigned picks marked) in OUT_DIR/unmatched/, for checking by eye.

    python g1_compare.py OUT_DIR TAG START END      e.g. p0.3_s0.3 2026-09-01 2026-10-01
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.spatial import Delaunay

from g1_common import BOX, DEG_KM, day_stream, stations


def epi_km(lat1, lon1, lat2, lon2):
    dx = (lon2 - lon1) * DEG_KM * np.cos(np.radians((lat1 + lat2) / 2))
    return np.hypot(dx, (lat2 - lat1) * DEG_KM)


def catalog(start, end):
    cat = Client("USGS").get_events(starttime=start, endtime=end, **BOX)
    rows = []
    for ev in cat:
        o = ev.preferred_origin() or ev.origins[0]
        m = ev.preferred_magnitude() or ev.magnitudes[0]
        rows.append(dict(id=str(ev.resource_id).split("eventid=")[-1].split("&")[0],
                         time=o.time.timestamp, latitude=o.latitude, longitude=o.longitude,
                         depth=o.depth / 1000, mag=m.mag,
                         etype=ev.event_type or ""))
    return pd.DataFrame(rows)


def match(cat, ours):
    pairs = []
    for i, c in cat.iterrows():
        dt = (ours["time"] - c["time"]).abs()
        d = epi_km(c["latitude"], c["longitude"], ours["latitude"], ours["longitude"])
        for j in ours.index[(dt <= 3) & (d <= 15)]:
            pairs.append((dt[j], i, j))
    used_c, used_o, out = set(), set(), []
    for dt, i, j in sorted(pairs):
        if i not in used_c and j not in used_o:
            used_c.add(i); used_o.add(j); out.append((i, j))
    return out


def plot_unmatched(out, tag, ours, assign, sta, unmatched):
    pdir = out / f"unmatched_{tag}"
    pdir.mkdir(exist_ok=True)
    stream, stream_day = None, None
    for j in unmatched:
        e = ours.loc[j]
        t0 = UTCDateTime(e["time"])
        day = t0.strftime("%Y%m%d")
        if day != stream_day:
            stream, stream_day = day_stream(out, day), day
        st = stream.slice(t0 - 5, t0 + 35).select(channel="?HZ").copy()
        st.detrend("demean").filter("bandpass", freqmin=2, freqmax=20)
        dist = {r.id: epi_km(e["latitude"], e["longitude"], r.latitude, r.longitude)
                for r in sta.itertuples()}
        st.traces.sort(key=lambda tr: dist.get(f"{tr.stats.network}.{tr.stats.station}", 1e9))
        st = st[:12]
        ep = assign[assign["event"] == e["event"]]
        fig, ax = plt.subplots(figsize=(9, 0.55 * max(len(st), 1) + 1.2))
        for k, tr in enumerate(st):
            sid = f"{tr.stats.network}.{tr.stats.station}"
            t = tr.times() + (tr.stats.starttime - t0)
            y = tr.data / (np.abs(tr.data).max() or 1)
            ax.plot(t, y * 0.45 - k, "k", lw=0.5)
            ax.text(-5.2, -k, f"{tr.stats.station} {dist.get(sid, 0):.0f} km", ha="right",
                    va="center", fontsize=7)
            for p in ep[ep["station"] == sid].itertuples():
                ax.vlines(p.time_raw - t0.timestamp, -k - 0.45, -k + 0.45,
                          color="tab:blue" if p.phase == "P" else "tab:red", lw=1.2)
        ax.set_xlim(-5, 35); ax.set_yticks([]); ax.set_xlabel("seconds after our origin time")
        ax.set_title(f"{str(t0)[:21]}  {e['latitude']:.3f} {e['longitude']:.3f} "
                     f"z={e['depth']:.1f} km  P={e['n_p']} S={e['n_s']} rms={e['rms']:.2f}s  "
                     f"(blue P, red S, 2–20 Hz)", fontsize=8)
        fig.tight_layout()
        fig.savefig(pdir / f"{t0.strftime('%Y%m%dT%H%M%S')}.png", dpi=90)
        plt.close(fig)


def main():
    out, tag = Path(sys.argv[1]), sys.argv[2]
    start, end = UTCDateTime(sys.argv[3]), UTCDateTime(sys.argv[4])
    ours = pd.read_csv(out / f"events_{tag}.csv")
    ours = ours[(ours["time"] >= start.timestamp) & (ours["time"] < end.timestamp)].reset_index(drop=True)
    assign = pd.read_csv(out / f"assignments_{tag}.csv")
    sta = stations(out)
    cat = catalog(start, end)
    hull = Delaunay(sta[["longitude", "latitude"]].to_numpy())
    cat["interior"] = hull.find_simplex(cat[["longitude", "latitude"]].to_numpy()) >= 0

    pairs = match(cat, ours)
    cat["matched"] = False
    cat.loc[[i for i, _ in pairs], "matched"] = True
    m = pd.DataFrame([dict(cat_id=cat.at[i, "id"], mag=cat.at[i, "mag"], interior=cat.at[i, "interior"],
                           dt=ours.at[j, "time"] - cat.at[i, "time"],
                           epi_km=epi_km(cat.at[i, "latitude"], cat.at[i, "longitude"],
                                         ours.at[j, "latitude"], ours.at[j, "longitude"]),
                           dz_km=ours.at[j, "depth"] - cat.at[i, "depth"]) for i, j in pairs])
    inner = cat[cat["interior"]]
    print(f"catalog: {len(cat)} events in box, {len(inner)} inside the network "
          f"(types: {cat['etype'].value_counts().to_dict()})")
    print(f"ours: {len(ours)} events")
    print(f"recall inside network: {inner['matched'].sum()}/{len(inner)} = {inner['matched'].mean():.0%}")
    print(f"recall whole box:      {cat['matched'].sum()}/{len(cat)} = {cat['matched'].mean():.0%}")
    bins = [-9, 1.0, 1.5, 2.0, 9]
    cat["mbin"] = pd.cut(cat["mag"], bins, labels=["<1.0", "1.0-1.5", "1.5-2.0", ">=2.0"])
    print("recall inside network by magnitude:")
    print(cat[cat["interior"]].groupby("mbin", observed=False)["matched"].agg(["sum", "count"]).to_string())
    if len(m):
        mi = m[m["interior"]]
        print(f"location vs catalog (interior, n={len(mi)}): median epicentre {mi['epi_km'].median():.2f} km, "
              f"90th pct {mi['epi_km'].quantile(0.9):.2f} km, median |depth diff| {mi['dz_km'].abs().median():.2f} km, "
              f"median depth bias {mi['dz_km'].median():+.2f} km, median |dt| {mi['dt'].abs().median():.2f} s")
    unmatched = [j for j in ours.index if j not in {j for _, j in pairs}]
    print(f"our events with no catalog match: {len(unmatched)}")
    cat.to_csv(out / f"compare_catalog_{tag}.csv", index=False)
    m.to_csv(out / f"compare_matches_{tag}.csv", index=False)
    ours.loc[unmatched].to_csv(out / f"compare_unmatched_{tag}.csv", index=False)
    plot_unmatched(out, tag, ours, assign, sta, unmatched)


if __name__ == "__main__":
    main()
