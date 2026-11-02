"""Smoke test: pretrained PhaseNet on one day of New Madrid data vs the USGS catalog."""
import sys, time, numpy as np, torch
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
import seisbench.models as sbm
day = UTCDateTime(sys.argv[1]); out = sys.argv[2]
stas = "CACT,CATM,FPST,GLAT,HALT,HENM,LNXT,MARM,MIST,NWCT,PARM,PEBM,PENM,PVMO,RDGT,STAM2,HICK,GNAR"
c = Client(base_url="https://service.earthscope.org")
t0 = time.time()
st = Stream()
for s in stas.split(","):
    try: st += c.get_waveforms("NM", s, "*", "HH?", day, day + 86400)
    except Exception as e: print("miss", s, type(e).__name__)
st.merge(fill_value=0)
print(f"downloaded {len(st)} traces from {len({t.stats.station for t in st})} stations in {time.time()-t0:.0f}s, "
      f"{sum(t.data.nbytes for t in st)/1e6:.0f} MB in memory")
model = sbm.PhaseNet.from_pretrained("instance").to("cuda")
t0 = time.time()
picks = model.classify(st, batch_size=512, P_threshold=0.3, S_threshold=0.3).picks
print(f"picked in {time.time()-t0:.0f}s: {sum(p.phase=='P' for p in picks)} P, {sum(p.phase=='S' for p in picks)} S")
# crude association: P picks on >=4 distinct stations within 6 s
P = sorted((p.peak_time, p.trace_id.split('.')[1]) for p in picks if p.phase == "P")
events, i = [], 0
while i < len(P):
    j = i
    while j + 1 < len(P) and P[j + 1][0] - P[i][0] <= 6: j += 1
    sts = {s for _, s in P[i:j + 1]}
    if len(sts) >= 4: events.append((P[i][0], len(sts))); i = j + 1
    else: i += 1
print(f"candidate events (P on >=4 stations within 6 s): {len(events)}")
cat = c if False else Client("USGS").get_events(starttime=day, endtime=day + 86400, minlatitude=35.5, maxlatitude=37.2,
                                                 minlongitude=-90.5, maxlongitude=-88.8)
found = 0
for ev in cat:
    o = ev.preferred_origin() or ev.origins[0]; m = ev.preferred_magnitude() or ev.magnitudes[0]
    hit = [e for e in events if -2 <= e[0] - o.time <= 15]
    found += bool(hit)
    print(f"  catalog {o.time} M{m.mag:.2f} -> {'FOUND' if hit else 'missed'} {hit[0][1] if hit else ''}")
print(f"catalog events {len(cat)}, found {found}; extra candidates {len(events) - found}")
with open(out, "w") as f:
    for t, n in events: f.write(f"{t} {n}\n")
