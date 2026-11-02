"""Start a command detached from this terminal and print the CHILD's pid.

    python scripts/launch_bg.py <logfile> -- <command> [args...]

`setsid nohup cmd &` reports the pid of a short-lived setsid parent when the
shell is a process-group leader, so status checks keyed on `$!` report a
false exit (this bit three launches in Sep 2026). Popen with
start_new_session=True gives the real pid, in its own session so it survives
the terminal, with stdin closed and stdout/stderr appended to the log.

For a pipeline run, launch `python -m insar_pipeline.pipeline <config> ...`
(or one of the scripts/run_*.py, which call the same thing): the run takes
the project lock, writes run.json and run_result.json itself, so it shows
in the dashboard, can be stopped there, and a second start on the same
project is refused (exit 3) - nothing about that depends on how it was
launched. Output is unbuffered so the log (and run_result.json's log tail)
is current.
"""
import os
import subprocess
import sys
from pathlib import Path

if "--" not in sys.argv or sys.argv.index("--") < 2:
    sys.exit(__doc__)
log = Path(sys.argv[1]); cmd = sys.argv[sys.argv.index("--") + 1:]
log.parent.mkdir(parents=True, exist_ok=True)
with open(log, "a") as fh:
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=str(Path.home()),
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
print(p.pid)
