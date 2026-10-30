"""Gate 1, step 1: download a month of New Madrid seismograph data and pick P/S phases.

For each day: fetch the station set's channels for the stations in the box (EarthScope FDSN),
keep the day as miniSEED (so later gates can re-pick without downloading again), run pretrained
PhaseNet on the GPU, and write every pick above a low threshold with its probability. Thresholds
are applied later, at association time.

Station sets:
    HH     NM broadband (the first set; files are mseed/, picks/, stations.csv)
    extra  every other HH or EH station in the box: NM short-period, KY, 6P, AO
           (files are mseed_extra/, picks_extra/, stations_extra.csv)

    python g1_pick.py 2026-09-01 2026-10-01 OUT_DIR [SET]
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import seisbench.models as sbm
from obspy import Stream, UTCDateTime, read
from obspy.clients.fdsn import Client

from g1_common import BOX

PICK_THRESHOLD = 0.1
SUFFIX = {"HH": "", "extra": "_extra"}


def stations(client, start, end, out, which):
    inv = client.get_stations(network="*", channel="HH?,EH?", starttime=start, endtime=end,
                              level="channel", **BOX)
    # A station can appear once per epoch; pool its bands across epochs.
    sites, bands = {}, {}
    for n in inv:
        for s in n:
            sid = f"{n.code}.{s.code}"
            sites.setdefault(sid, s)
            bands.setdefault(sid, set()).update(c.code[:2] for c in s.channels)
    rows = []
    for sid, s in sites.items():
        band = "HH" if "HH" in bands[sid] else "EH"  # broadband where a site has both
        first = sid.startswith("NM.") and band == "HH"
        if first == (which == "HH"):
            rows.append(dict(id=sid, latitude=s.latitude, longitude=s.longitude,
                             elevation=s.elevation, band=band))
    df = pd.DataFrame(rows)
    if which != "HH" and (out / "stations.csv").exists():  # never the same site in two sets
        df = df[~df["id"].isin(pd.read_csv(out / "stations.csv")["id"])]
    df.to_csv(out / f"stations{SUFFIX[which]}.csv", index=False)
    return df


def fetch(client, sid, band, t0):
    net, sta = sid.split(".")
    try:
        # EH? would also match oddities like EHG; ask for the ground-motion components only.
        chans = ",".join(band + c for c in "ZNE12")
        return client.get_waveforms(net, sta, "*", chans, t0, t0 + 86400)
    except Exception:
        return Stream()


def main():
    start, end, out = UTCDateTime(sys.argv[1]), UTCDateTime(sys.argv[2]), Path(sys.argv[3])
    which = sys.argv[4] if len(sys.argv) > 4 else "HH"
    mseed_dir, pick_dir = out / f"mseed{SUFFIX[which]}", out / f"picks{SUFFIX[which]}"
    mseed_dir.mkdir(parents=True, exist_ok=True)
    pick_dir.mkdir(exist_ok=True)
    client = Client(base_url="https://service.earthscope.org")
    sta = stations(client, start, end, out, which)
    print(f"{len(sta)} stations", flush=True)
    model = sbm.PhaseNet.from_pretrained("instance").to("cuda")
    day = start
    while day < end:
        tag = day.strftime("%Y%m%d")
        pick_file = pick_dir / f"{tag}.csv"
        if pick_file.exists():
            day += 86400
            continue
        t0 = time.time()
        mseed = mseed_dir / f"{tag}.mseed"
        if mseed.exists():
            st = read(str(mseed))
        else:
            with ThreadPoolExecutor(4) as pool:
                parts = pool.map(lambda r: fetch(client, r.id, r.band, day), sta.itertuples())
            st = Stream()
            for p in parts:
                st += p
            st.write(str(mseed), format="MSEED")
        t_dl = time.time() - t0
        st.merge(fill_value=0)
        picks = model.classify(st, batch_size=512, P_threshold=PICK_THRESHOLD,
                               S_threshold=PICK_THRESHOLD).picks
        rows = [dict(station=".".join(p.trace_id.split(".")[:2]), phase=p.phase,
                     time=p.peak_time.timestamp, prob=p.peak_value) for p in picks]
        pd.DataFrame(rows, columns=["station", "phase", "time", "prob"]).to_csv(pick_file, index=False)
        n_sta = len({t.stats.station for t in st})
        print(f"{tag}: {n_sta} stations, {len(rows)} picks, download {t_dl:.0f}s, "
              f"total {time.time() - t0:.0f}s", flush=True)
        day += 86400


if __name__ == "__main__":
    main()
