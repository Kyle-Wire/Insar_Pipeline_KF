"""Test-wide isolation (16 Oct 2026, readiness review ERR-05): no test
writes into the real ~/.insar_pipeline_notifications.log or raises a Windows
toast. Before this, modules that forgot to stub notifications (and pipeline
runs in subprocesses, which no monkeypatch reaches) left lines such as
"InSAR proj: FAILED" in the operator's real log. The environment variables
reach subprocesses; LOGFILE covers notify modules already imported. A test
that needs its own log or toasts still sets them itself.

The same for the run registry (16 Oct 2026, readiness review OUT-05):
every pipeline run a test makes ends with a line in
<projects root>/run_history.jsonl, and the projects root defaults to the
real ~/insar_projects; INSAR_RUN_HISTORY sends it to the test's base
folder (a test that reads the registry sets its own)."""
import pytest


@pytest.fixture(autouse=True)
def _no_real_notifications(tmp_path_factory, monkeypatch):
    log = tmp_path_factory.getbasetemp() / "notifications.log"
    monkeypatch.setenv("INSAR_NOTIFY", "off")
    monkeypatch.setenv("INSAR_NOTIFY_LOG", str(log))
    monkeypatch.setenv("INSAR_RUN_HISTORY", str(tmp_path_factory.getbasetemp() / "run_history.jsonl"))
    from insar_pipeline import notify
    monkeypatch.setattr(notify, "LOGFILE", log)
