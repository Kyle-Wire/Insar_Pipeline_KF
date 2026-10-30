"""Structured progress events for a run: <project>/progress.jsonl.

One JSON object per line, appended as work completes, so a dashboard can show
what is DONE on disk rather than guessing from log text:

    {"t": 1759000000.0, "stage": "disp_prefetch", "status": "running",
     "done": 41, "total": 254, "msg": "OPERA_L3_DISP-S1_..."}

status is one of start / running / done / failed / skipped. A stage's latest
event wins. A counted stage that re-verifies work a previous run finished
also reports how much of `done` was only verified ("skipped": 600 of done):
the remaining-time estimate is measured on the real work alone. Writes are single appends of one short line, which POSIX keeps
atomic for O_APPEND writes this small, so concurrent worker processes can
report without a lock. A failure to write never stops the pipeline.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

FILENAME = "progress.jsonl"
STATUSES = ("start", "running", "done", "failed", "skipped")


def path_for(project_dir) -> Path:
    return Path(project_dir) / FILENAME


def emit(project_dir, stage: str, status: str = "running", done: int | None = None,
         total: int | None = None, msg: str = "", skipped: int | None = None, **extra) -> None:
    if status not in STATUSES:
        raise ValueError(f"status {status!r} is not one of {STATUSES}")
    rec = {"t": round(time.time(), 3), "stage": stage, "status": status}
    if done is not None:
        rec["done"] = int(done)
    if total is not None:
        rec["total"] = int(total)
    if skipped is not None:
        rec["skipped"] = int(skipped)
    if msg:
        rec["msg"] = str(msg)[:300]
    rec.update(extra)
    try:
        p = path_for(project_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps(rec, default=str) + "\n").encode())
        finally:
            os.close(fd)
    except OSError:
        pass


def read(project_dir) -> list[dict]:
    """Every event, oldest first; unparseable lines (a torn write) skipped."""
    p = path_for(project_dir)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def latest(project_dir) -> dict[str, dict]:
    """The most recent event per stage, in the order stages first appeared.

    A stage's first event time is kept as "started" so elapsed time and a rate
    can be computed from the latest one. "work_started" is the time of the
    stage's last event at which nothing had yet been done for real (done ==
    skipped): the clock for the rate of the real work."""
    stages: dict[str, dict] = {}
    for ev in read(project_dir):
        s = ev.get("stage")
        if not s or "t" not in ev:
            continue
        prev = stages.get(s, {})
        first = prev.get("started", ev["t"])
        work0 = prev.get("work_started", first)
        # a new "start" after a finished stage is a re-run: restart the clock
        if ev.get("status") == "start":
            first = work0 = ev["t"]
        elif ev.get("done") is not None and int(ev.get("done") or 0) <= int(ev.get("skipped") or 0):
            work0 = ev["t"]          # still only verifying the previous run's work
        stages[s] = {**prev, **ev, "started": first, "work_started": work0}
        if ev.get("status") == "start":
            for k in ("done", "total", "msg", "skipped"):
                if k not in ev:
                    stages[s].pop(k, None)
    return stages


def eta_seconds(ev: dict, now: float | None = None) -> float | None:
    """Remaining time for a counted stage from its own measured rate.

    Only REAL work counts: commands a resumed stage verified on disk
    ("skipped") finish in milliseconds, and counting them as done made the
    estimate collapse - 600 of 804 verified in 2 s, then 10 real unwraps in
    120 s read as 39 s left against ~40 min. The rate is real items over the
    time since the verifying ended (work_started). With `now`, the estimate
    counts down between events instead of standing still at the last one;
    callers pass it only while the run is alive (a vanished run has no ETA)."""
    done, total = ev.get("done"), ev.get("total")
    if not done or not total or done >= total or ev.get("status") != "running":
        return None
    real = done - int(ev.get("skipped") or 0)
    if real <= 0:
        return None
    t0 = ev.get("started", ev["t"])
    if ev.get("skipped"):
        t0 = ev.get("work_started", t0)
    elapsed = ev["t"] - t0
    if elapsed <= 0:
        return None
    left = elapsed / real * (total - done)
    if now is not None:
        left = max(0.0, left - max(0.0, now - ev["t"]))
    return left
