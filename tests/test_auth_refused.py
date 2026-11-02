"""A download timeout is not a refused login (22 Oct 2026): a CMR timeout's
traceback carried 'line 403' and stopped a 274-date download as 'login refused'."""
from insar_pipeline import data_acquisition as da


def test_traceback_line_numbers_and_timeouts_are_not_a_refused_login():
    tb = ('  File ".../tenacity/__init__.py", line 403, in __call__\n'
          '  File ".../search_generator.py", line 385, in get_page\n'
          'asf_search.exceptions.ASFSearchError: Connection Error (Timeout): CMR took too long to respond.')
    assert not da.auth_refused(tb)
    assert not da.auth_refused('  File "x.py", line 401, in f\nValueError: boom')


def test_real_refusals_still_count():
    assert da.auth_refused("requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://urs")
    assert da.auth_refused("HTTP 403 Forbidden")
    assert da.auth_refused("authentication failed for user")


def test_cmr_search_retries_a_timeout_with_a_longer_limit(monkeypatch):
    """22 Oct 2026: three launches died on asf_search's 30 s CMR timeout before
    downloading anything. The search now waits 120 s and retries timeouts."""
    import asf_search.constants.INTERNAL as asf_internal
    calls, slept = [], []

    def flaky(**opts):
        calls.append(opts)
        if len(calls) < 3:
            raise RuntimeError("Connection Error (Timeout): CMR took too long to respond.")
        return ["result"]

    monkeypatch.setattr(da.asf, "search", flaky)
    monkeypatch.setattr(da.time, "sleep", slept.append)
    monkeypatch.setattr(asf_internal, "CMR_TIMEOUT", 30)
    assert da.cmr_search(dataset="X") == ["result"]
    assert len(calls) == 3 and slept == [30.0, 60.0]
    assert asf_internal.CMR_TIMEOUT == da.CMR_TIMEOUT_S


def test_cmr_search_raises_other_errors_at_once(monkeypatch):
    import pytest
    calls = []

    def broken(**opts):
        calls.append(opts)
        raise ValueError("bad polygon")

    monkeypatch.setattr(da.asf, "search", broken)
    monkeypatch.setattr(da.time, "sleep", lambda s: None)
    with pytest.raises(ValueError):
        da.cmr_search(dataset="X")
    assert len(calls) == 1


def test_cmr_search_gives_up_after_the_last_retry(monkeypatch):
    import pytest

    def always(**opts):
        raise RuntimeError("Connection Error (Timeout): CMR took too long to respond.")

    monkeypatch.setattr(da.asf, "search", always)
    monkeypatch.setattr(da.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        da.cmr_search(dataset="X")
