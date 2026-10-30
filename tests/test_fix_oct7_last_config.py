"""7 Oct 2026, last pass, group config: decision M3 (a reissue says so).

The Site A report rebuilt on 7 Oct replaces the one issued on 28 Sep
2026 with a different headline and different figures, and nothing on the
page, the PDF or the KMZ said so; the release gate did not ask for a note.
Now:
  - config.py: a report_revision setting ({number, supersedes, changes}),
    checked when the config loads, and config.report_revision(), the one
    wording ("Revision 1, replacing the report issued 28 Sep 2026") the
    page, the PDF, the KMZ and the gate share; a setting given twice in a
    YAML is refused (PyYAML kept the second silently, which is how a second
    note written under the first replaced it on 7 Oct);
  - release.py: the hard blocker 'reissue_without_note' when the project has
    already issued a report with other numbers (deliverables/final_*, or a
    final package in deliverables/releases/) and this one carries no note,
    or a note older than the latest issue;
  - configs/config_site_a_pl.yaml: Revision 1 of the 28 Sep report, its
    "before" figures checked here against the delivered record.

Each test failed on the code in backups/pre_final_20261007/ unless its
docstring says otherwise. Synthetic projects in tmp_path, plus read-only
reads of the delivered Site A record."""
import json
import re
import textwrap
from pathlib import Path

import pytest

import test_fix_L7b as L7b                                     # the gate's synthetic project
from insar_pipeline import config as C
from insar_pipeline import package, release

PKG = Path(__file__).resolve().parents[1]
IN = 25.4

NOTE = {"number": 1, "supersedes": "2026-09-28",
        "changes": ["Every figure is now compared with farmland that leaves out the sinking area.",
                    "Each total over the record now carries its 95% likely range."]}


@pytest.fixture(autouse=True)
def _identity_set(monkeypatch):
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})


def _yaml(tmp_path, extra: str) -> Path:
    y = tmp_path / "c.yaml"
    y.write_text(f"project_dir: {tmp_path / 'p'}\nbbox: [-87.8, 38.2, -87.52, 38.47]\nstart_date: '2018-06-01'\n"
                 "end_date: '2019-06-30'\nbackend: hyp3\n" + textwrap.dedent(extra))
    return y


def _codes(b, hard=None):
    return [x["code"] for x in b if hard is None or x["hard"] is hard]


def _issued(tmp_path, folder="final_20260928", rep=None, manifest=None) -> Path:
    """An earlier issue: a hand-assembled final_* folder or a package, with its
    own client_report.json (other numbers unless rep is given)."""
    d = tmp_path / "deliverables" / folder
    (d / "technical").mkdir(parents=True)
    old = rep or L7b._rep()
    if rep is None:
        old["areas"][1]["vertical_mm_yr"] = 0.17
    (d / "technical" / "client_report.json").write_text(json.dumps(old))
    if manifest is not None:
        (d / "MANIFEST.json").write_text(json.dumps(manifest))
    return d


def _signed(tmp_path, cfg=None, rep=None):
    cfg = cfg or L7b._cfg(tmp_path)
    rep = rep or L7b._rep()
    L7b._project(tmp_path, rep, L7b._qa())
    L7b._sign_all(tmp_path, rep)
    return cfg, rep


# ------------------------------------------------------------------ config.py: the setting
def test_a_revision_note_loads_and_reads_as_one_heading(tmp_path):
    """The YAML block, with the date unquoted (YAML reads it as a date), loads;
    report_revision() gives the heading the page, PDF, KMZ and gate share,
    and reads its own output back unchanged (the report may carry it)."""
    cfg = C.load_config(str(_yaml(tmp_path, """
        report_revision:
          number: 1
          supersedes: 2026-09-28
          changes:
            - "Detection limits   now at 80% power."
            - "Totals now carry their 95% range."
        """)))
    assert cfg.report_revision["supersedes"] == "2026-09-28"
    r = C.report_revision(cfg)
    assert r["heading"] == "Revision 1, replacing the report issued 28 Sep 2026"
    assert r["number"] == 1 and r["supersedes"] == "2026-09-28" and r["supersedes_text"] == "28 Sep 2026"
    assert r["changes"] == ["Detection limits now at 80% power.", "Totals now carry their 95% range."]
    assert C.report_revision(r) == r
    from insar_pipeline import provenance
    assert "report_revision" in provenance.config_snapshot(cfg)[0]


def test_no_revision_note_is_a_first_issue(tmp_path):
    """Left out, or written with nothing under it: a first issue, no heading."""
    assert C.report_revision(C.load_config(str(_yaml(tmp_path, "")))) is None
    cfg = C.load_config(str(_yaml(tmp_path, "report_revision:\n")))
    assert cfg.report_revision == {} and C.report_revision(cfg) is None
    assert C.report_revision(None) is None and C.report_revision({}) is None


@pytest.mark.parametrize("block, says", [
    ("  number: 1\n  supersedes: '2026-09-28'\n", "list of plain sentences"),
    ("  number: 1\n  supersedes: '2026-09-28'\n  changes: []\n", "list of plain sentences"),
    ("  number: 1\n  supersedes: '2026-09-28'\n  changes: ['  ']\n", "none may be empty"),
    ("  number: 0\n  supersedes: '2026-09-28'\n  changes: ['x']\n", "must be a whole number"),
    ("  number: one\n  supersedes: '2026-09-28'\n  changes: ['x']\n", "must be a whole number"),
    ("  number: 1\n  supersedes: '28/09/2026'\n  changes: ['x']\n", "as YYYY-MM-DD"),
    ("  number: 1\n  changes: ['x']\n", "as YYYY-MM-DD"),
    ("  number: 1\n  supersedes: '2099-01-01'\n  changes: ['x']\n", "is in the future"),
    ("  number: 1\n  supersedes: '2026-09-28'\n  changes: ['x']\n  date: '2026-10-08'\n", "unknown key(s) ['date']"),
])
def test_a_half_written_revision_note_is_refused(tmp_path, block, says):
    """A note the page would print as a heading with no changes (or the gate
    would read as no note) stops the config from loading, in plain words."""
    with pytest.raises(C.ConfigError, match=re.escape(says)):
        C.load_config(str(_yaml(tmp_path, "report_revision:\n" + block)))
    assert C.report_revision({"number": 1, "supersedes": "2026-09-28"}) is None


def test_a_setting_given_twice_is_refused(tmp_path):
    """On 7 Oct 2026 a second report_revision block written below the first
    in config_site_a_pl.yaml replaced it without a word: PyYAML keeps the
    last of two equal keys. The loader now names both lines."""
    two = ("report_revision:\n  number: 1\n  supersedes: '2026-09-28'\n  changes: ['first']\n"
           "report_revision:\n  number: 1\n  supersedes: '2026-09-28'\n  changes: ['second']\n")
    with pytest.raises(C.ConfigError, match=r"'report_revision' is given twice \(lines 6 and 10\)"):
        C.load_config(str(_yaml(tmp_path, two)))
    with pytest.raises(C.ConfigError, match="'end_date' is given twice"):
        C.load_config(str(_yaml(tmp_path, "end_date: '2019-05-30'\n")))
    with pytest.raises(C.ConfigError, match="'number' is given twice"):
        C.load_config(str(_yaml(tmp_path, "report_revision:\n  number: 1\n  number: 2\n")))


# ------------------------------------------------------------------ release.py: the gate
def test_a_reissue_without_a_note_is_a_hard_blocker(tmp_path):
    """Site A's case: a signed report with every other check clear, in a
    project that issued final_20260928/ with other numbers."""
    cfg, rep = _signed(tmp_path)
    assert release.is_final(cfg)                                           # a first issue needs no note
    _issued(tmp_path)
    b = release.blockers(cfg)
    assert _codes(b, hard=True) == ["reissue_without_note"], b
    t = b[[x["code"] for x in b].index("reissue_without_note")]["text"]
    assert "replaces the one issued 28 Sep 2026 (deliverables/final_20260928)" in t
    assert "set report_revision" in t and "the page, the PDF and the KMZ" in t
    assert release.is_final(cfg) is False and release.status(cfg)["final"] is False
    # the note in the settings settles it
    cfg.report_revision = dict(NOTE)
    assert "reissue_without_note" not in _codes(release.blockers(cfg))
    assert release.is_final(cfg)


def test_the_gate_reads_the_note_the_report_printed(tmp_path):
    """A report that records it was built with no note (report_revision None)
    did not print one, whatever the settings say now; a report carrying the
    note passes even for a caller with only the project folder (the
    dashboard's projects table)."""
    cfg = L7b._cfg(tmp_path)
    cfg.report_revision = dict(NOTE)
    built_before = L7b._rep(report_revision=None)
    _signed(tmp_path, cfg, built_before)
    _issued(tmp_path)
    t = next(x["text"] for x in release.hard_blockers(cfg) if x["code"] == "reissue_without_note")
    assert "built before the note was set" in t and "Revision 1, replacing the report issued 28 Sep 2026" in t
    carrying = L7b._rep(report_revision=C.report_revision(NOTE))
    (tmp_path / "deliverables" / "client_report.json").write_text(json.dumps(carrying))
    L7b._sign_all(tmp_path, carrying)
    assert release.is_final(cfg)
    assert "reissue_without_note" not in _codes(release.blockers(tmp_path))
    # a note edited in the settings after the build is not the one the page prints
    cfg.report_revision = dict(NOTE, changes=["A corrected sentence."])
    t = next(x["text"] for x in release.hard_blockers(cfg) if x["code"] == "reissue_without_note")
    assert "changed after the report was built" in t
    cfg.report_revision = dict(NOTE)
    assert release.is_final(cfg)
    # the note is part of the signed numbers: a changed sentence voids the sign-off
    edited = json.loads(json.dumps(carrying))
    edited["report_revision"]["changes"][0] = "Something else."
    (tmp_path / "deliverables" / "client_report.json").write_text(json.dumps(edited))
    assert "signoff_stale" in _codes(release.blockers(cfg))


def test_a_folder_only_caller_reads_the_config_snapshot(tmp_path):
    """The dashboard asks with the project folder alone; for a report built
    before reports carried the note, the settings the products run used
    (deliverables/config_snapshot.yaml) say whether there is one."""
    _signed(tmp_path)
    _issued(tmp_path)
    assert "reissue_without_note" in _codes(release.blockers(tmp_path), hard=True)
    import yaml
    snap = tmp_path / "deliverables" / "config_snapshot.yaml"
    snap.write_text(yaml.safe_dump({"backend": "miaplpy", "report_revision": dict(NOTE)}))
    assert "reissue_without_note" not in _codes(release.blockers(tmp_path))
    snap.write_text(yaml.safe_dump({"backend": "miaplpy", "report_revision": {}}))
    assert "reissue_without_note" in _codes(release.blockers(tmp_path), hard=True)


def test_a_note_older_than_the_latest_issue_is_a_hard_blocker(tmp_path):
    """Revision 2 after a packaged Revision 1: a note still replacing the
    28 Sep issue does not explain what changed since 1 Oct."""
    cfg, rep = _signed(tmp_path)
    cfg.report_revision = dict(NOTE)
    _issued(tmp_path)
    rev1 = L7b._rep()
    rev1["areas"][1]["vertical_mm_yr"] = -0.56
    _issued(tmp_path, "releases/20261001_1200", rev1,
            {"final": True, "created": "2026-10-01T12:00:00", "numbers_hash": release.numbers_hash(rev1)})
    assert [i["date"] for i in release.earlier_issues(tmp_path, rep)] == ["2026-09-28", "2026-10-01"]
    t = next(x["text"] for x in release.hard_blockers(cfg) if x["code"] == "reissue_without_note")
    assert "replaces the one issued 28 Sep 2026, but a later report was issued 1 Oct 2026" in t
    cfg.report_revision = {"number": 2, "supersedes": "2026-10-01", "changes": ["What moved since Revision 1."]}
    assert release.is_final(cfg)


def test_drafts_partial_copies_and_this_reports_own_package_are_not_earlier_issues(tmp_path):
    """Only an issue with other numbers makes a reissue: a _DRAFT package, a
    package being written (.name.partial), one whose manifest says it is not
    final, and a package of these very numbers are not."""
    cfg, rep = _signed(tmp_path)
    moved = L7b._rep()
    moved["areas"][1]["vertical_mm_yr"] = 3.0
    _issued(tmp_path, "releases/20261008_0900_DRAFT", moved, {"final": False})
    _issued(tmp_path, "releases/.20261008_1000.partial", moved)
    _issued(tmp_path, "releases/20261008_1100", moved, {"final": False})
    _issued(tmp_path, "releases/20261008_1200", rep, {"final": True, "numbers_hash": release.numbers_hash(rep)})
    assert release.earlier_issues(tmp_path, rep) == []
    assert release.is_final(cfg)
    # an issued copy whose report cannot be read is still an issue
    _issued(tmp_path, "final_20260101", moved)
    (tmp_path / "deliverables" / "final_20260101" / "technical" / "client_report.json").write_text("{not json")
    assert [i["label"] for i in release.earlier_issues(tmp_path, rep)] == ["final_20260101"]


def test_a_real_package_then_new_numbers_needs_a_note(tmp_path):
    """The whole flow: package a signed report (its own package is not an
    earlier issue), rebuild with other numbers and sign again - the package
    is now the issue the new report replaces."""
    cfg, rep = _signed(tmp_path)
    out = package.build(cfg)
    assert release.is_final(cfg) and release.earlier_issues(tmp_path, rep) == []
    moved = json.loads(json.dumps(rep))
    moved["areas"][1]["vertical_mm_yr"] = -0.56
    (tmp_path / "deliverables" / "client_report.json").write_text(json.dumps(moved))
    L7b._sign_all(tmp_path, moved)
    hard = release.hard_blockers(cfg)
    assert _codes(hard) == ["reissue_without_note"], hard
    assert f"(deliverables/releases/{out.name})" in hard[0]["text"]
    cfg.report_revision = {"number": 1, "supersedes": out.name[:4] + "-" + out.name[4:6] + "-" + out.name[6:8],
                           "changes": ["Area A1 now reads -0.02 in/yr."]}
    assert release.is_final(cfg)


# ------------------------------------------------------------------ independent review (7 Oct 2026)
def test_a_yaml_merge_still_loads_and_a_repeat_on_one_line_is_refused(tmp_path):
    """The duplicate-key loader must read what yaml.safe_load read: a merge
    ('<<: *anchor') failed with "could not determine a constructor for the
    tag 'tag:yaml.org,2002:merge'" in the first version of this pass (the
    old safe_load read it, so that half passes on the backup code). A key
    repeated inside one flow block ({a: 1, a: 2}, on one line) slipped past
    the line-number comparison; the old safe_load kept the second silently."""
    import yaml
    text = "a: &d {number: 1, supersedes: '2026-09-28'}\nb:\n  <<: *d\n  number: 2\n"
    merged = yaml.load(text, Loader=getattr(C, "_UniqueKeyLoader", yaml.SafeLoader))
    assert merged == yaml.safe_load(text) and merged["b"] == {"number": 2, "supersedes": "2026-09-28"}
    with pytest.raises(C.ConfigError, match=r"'number' is given twice \(line 6\)"):
        C.load_config(str(_yaml(tmp_path, "report_revision: {number: 1, number: 2, supersedes: '2026-09-28', "
                                          "changes: ['x']}\n")))


