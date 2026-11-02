"""QC map (8 Oct 2026, T1): the digitised panels, the Tahmoor West GNSS units,
the protocol A stations, the APPROX envelopes and the new deforming polygon,
in lon/lat. Writes ../checks/qc_map.png."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
TRUTH = HERE.parent
PKG = TRUTH.parents[1]
CTX = PKG / "configs" / "experiments" / "context"
STATIONS = {"CA19": (150.691241, -34.178878), "CA07": (150.6392, -34.2216), "PCTN": (150.6052, -34.1576),
            "MENA": (150.7438, -34.1261)}


def rings(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [p[0] for p in geom["coordinates"]]


def main():
    fig, ax = plt.subplots(figsize=(11, 9))
    for f in json.loads((CTX / "appin_tahmoor_deforming_APPROX.geojson").read_text())["features"]:
        for r in rings(f["geometry"]):
            x, y = zip(*r)
            ax.plot(x, y, color="#999999", lw=1, ls="--")
    for f in json.loads((CTX / "appin_tahmoor_deforming.geojson").read_text())["features"]:
        for r in rings(f["geometry"]):
            x, y = zip(*r)
            ax.fill(x, y, color="#f2c14e", alpha=0.35, lw=0)
            ax.plot(x, y, color="#c98a00", lw=1)
    for f in json.loads((TRUTH / "panels.geojson").read_text())["features"]:
        p = f["properties"]
        x, y = zip(*f["geometry"]["coordinates"][0])
        col = "#444444" if p["in_window"] else "#bbbbbb"
        ax.plot(x, y, color=col, lw=1.2, ls="-" if p["in_window"] else ":")
        if p["id"] != "TW_W4_planned":
            ax.text(sum(x[:-1]) / 4, sum(y[:-1]) / 4, p["id"].split("_", 1)[1], fontsize=6, ha="center", va="center")
    sl = TRUTH / "survey_lines.geojson"
    if sl.exists():
        for f in json.loads(sl.read_text())["features"]:
            x, y = zip(*f["geometry"]["coordinates"])
            ax.plot(x, y, color="#8e44ad", lw=1.2)
    seen = set()
    for r in csv.DictReader(line for line in open(TRUTH / "tahmoor_west_gnss_units.csv") if not line.startswith("#")):
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        ax.plot(float(r["lon"]), float(r["lat"]), "o", ms=3, color="#1f6fd1")
    for k, (lo, la) in STATIONS.items():
        ax.plot(lo, la, "^", ms=7, color="#c0392b")
        ax.text(lo + 0.002, la + 0.002, k, fontsize=8, color="#c0392b")
    w, s, e, n = 150.55, -34.30, 150.78, -34.14
    ax.plot([w, e, e, w, w], [s, s, n, n, s], color="k", lw=0.8)
    ax.set_xlim(w - 0.02, e + 0.02)
    ax.set_ylim(s - 0.03, n + 0.02)
    ax.set_aspect(1 / 0.827)
    ax.set_title("T1: digitised panels (grey: mined or settling 2017-11..2022-12), deforming polygon (orange), "
                 "APPROX (dashed), survey lines (purple),\nTahmoor West GNSS units (blue), protocol A stations (red), A box (black)", fontsize=9)
    fig.tight_layout()
    fig.savefig(TRUTH / "checks" / "qc_map.png", dpi=130)


if __name__ == "__main__":
    main()
