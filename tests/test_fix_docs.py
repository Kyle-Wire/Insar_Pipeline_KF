"""Docs lane (L11, review 5 Oct 2026): the user-facing documents name only
things that exist, and the user guide follows the code.

Until 5 Oct 2026 the runbook told coworkers to copy a config that did not
exist (config_ridgecrest_ts.yaml), to run a module that did not exist
(insar_pipeline.screening_report), and the README's quick start used a
config that releases did not ship (usability-10). These tests read README.md,
INSAR_RUNBOOK.md, USER_GUIDE.md, install/README.md and THIRD_PARTY.md and
check that:
  * every `python -m insar_pipeline.X` names a module file in the package;
  * every `scripts/...` path exists;
  * every `insar <subcommand>` is one the dispatcher (dashboard.main) handles;
  * every repository path in code spans and links exists;
  * the user guide sits where the dashboard's `?` key looks, names every send
    rule value, every quality check, every restart kind and the real sign-off
    and package keys;
  * the technical reference's table of new settings names real config fields;
  * THIRD_PARTY.md quotes the data credits exactly as credits.py prints them.
"""
from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

PKG = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "TECHNICAL_REFERENCE.md", "INSAR_RUNBOOK.md", "USER_GUIDE.md", "install/README.md",
        "THIRD_PARTY.md")
# Folders of the repository. A path in a doc whose first part is one of these
# is a repository path; anything else (deliverables/, mintpy/, stack/ ...) is
# a path inside a project folder and is not checked here.
REPO_DIRS = {"install", "patches", "context", "configs", "scripts", "tests"}
REPO_PREFIXES = ("~/insar_pipeline/", "$INSAR_PIPELINE/", "/home/kylew/insar_pipeline/", "insar_pipeline/")
BARE_SUFFIXES = (".md", ".py", ".sh", ".toml")
SKIP_SEARCH = ("backups", "review_", "__pycache__", ".pytest_cache")
# Scripts of the tools the pipeline drives (ISCE2, MintPy, MiaplPy), named in
# the docs; they live in those packages, not here.
EXTERNAL = {"unwrap.py", "iono_tec.py", "topsApp.py", "stackSentinel.py", "smallbaselineApp.py",
            "miaplpyApp.py", "Stack.py"}
# repository paths that exist only on the machine that has the client files (.gitignore)
LOCAL_ONLY = {"tests/test_local_*.py"}


def _text(doc: str) -> str:
    return (PKG / doc).read_text()


def _code(doc: str) -> list[str]:
    """Inline code spans, fenced blocks and indented (4+ spaces) code lines."""
    out, fenced = [], False
    for line in _text(doc).splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or (line.startswith("    ") and not line.lstrip().startswith("|")):
            out.append(line.strip())
        out.extend(re.findall(r"`([^`\n]+)`", line))
    return out


def _links(doc: str) -> list[str]:
    return [t for t in re.findall(r"\]\(([^)\s]+)\)", _text(doc)) if not re.match(r"[a-z]+:", t)]


def _exists(rel: str) -> bool:
    rel = rel.rstrip("/")
    if any(c in rel for c in "*?["):
        return any(PKG.glob(rel))
    return (PKG / rel).exists()


def _clean(tok: str) -> str:
    tok = tok.strip("\"'()[]{},;")
    tok = re.sub(r":\d+(-\d+)?$", "", tok)          # file.py:11-12
    return tok.rstrip(".,:;")


def _repo_path(tok: str) -> str | None:
    """The repository-relative path a token names, or None when it is not a
    repository path."""
    for p in REPO_PREFIXES:
        if tok.startswith(p):
            return tok[len(p):]
    if tok.startswith(("~", "$", "/", "\\", "<", ".", "-", "http")):
        return None
    if "/" in tok:
        return tok if tok.split("/", 1)[0] in REPO_DIRS else None
    if tok.endswith(BARE_SUFFIXES) and re.fullmatch(r"[A-Za-z0-9_.\-]+", tok):
        return tok
    return None


def _found_anywhere(name: str) -> bool:
    return any(not any(s in str(p.relative_to(PKG)) for s in SKIP_SEARCH) for p in PKG.rglob(name))


def _dispatcher_words() -> set[str]:
    """Every literal dashboard.main compares the subcommand with."""
    tree = ast.parse((PKG / "dashboard.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    words = set()
    for node in ast.walk(main):
        if isinstance(node, ast.Compare):
            for c in ast.walk(node):
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    words.add(c.value)
    return words


# ---------------------------------------------------------------- existence
@pytest.mark.parametrize("doc", DOCS)
def test_document_exists(doc):
    assert (PKG / doc).is_file(), doc


@pytest.mark.parametrize("doc", DOCS)
def test_python_m_modules_exist(doc):
    missing = []
    for mod in re.findall(r"python3? -m insar_pipeline\.([A-Za-z0-9_.]+)", _text(doc)):
        parts = mod.rstrip(".").split(".")
        f = PKG.joinpath(*parts)
        if not (f.with_suffix(".py").is_file() or (f / "__init__.py").is_file()):
            missing.append(mod)
    assert not missing, f"{doc} names modules that do not exist: {missing}"


@pytest.mark.parametrize("doc", DOCS)
def test_scripts_paths_exist(doc):
    missing = []
    for m in re.findall(r"(?<![\w/])(?:~/insar_pipeline/|insar_pipeline/)?(scripts/[A-Za-z0-9_./*-]*)", _text(doc)):
        rel = _clean(m)
        if rel and not _exists(rel):
            missing.append(rel)
    assert not missing, f"{doc} names scripts that do not exist: {missing}"


@pytest.mark.parametrize("doc", DOCS)
def test_insar_subcommands_exist(doc):
    known = _dispatcher_words()
    assert {"doctor", "builder", "cleanup", "results"} <= known      # the parser found the dispatcher
    bad = []
    for code in _code(doc):
        for seg in re.split(r"&&|\|\||;|\|", code):
            seg = seg.strip()
            m = re.match(r"insar\s+(\S+)", seg)
            if not m:
                continue
            sub = m.group(1)
            if sub.startswith("<") or sub.endswith((".yaml", ".yml")) or sub.startswith("#"):
                continue                                  # `insar <config.yaml>` opens that project
            if sub not in known:
                bad.append(seg)
    assert not bad, f"{doc} uses insar subcommands the dispatcher does not handle: {bad}"


@pytest.mark.parametrize("doc", DOCS)
def test_repository_paths_exist(doc):
    missing = []
    tokens = [t for code in _code(doc) for t in re.split(r"\s+", code)] + _links(doc)
    for raw in tokens:
        tok = _clean(raw)
        rel = _repo_path(tok)
        if not rel or "<" in rel or "{" in rel or rel in EXTERNAL or rel in LOCAL_ONLY:
            continue
        if "/" not in rel and tok.endswith(BARE_SUFFIXES):
            ok = (PKG / rel).exists() or _found_anywhere(rel)
        else:
            ok = _exists(rel)
        if not ok:
            missing.append(tok)
    assert not missing, f"{doc} names repository files that do not exist: {sorted(set(missing))}"


def test_retired_names_are_gone():
    """The stale commands the review found must not come back."""
    for doc in ("README.md", "TECHNICAL_REFERENCE.md", "INSAR_RUNBOOK.md", "USER_GUIDE.md", "install/README.md"):
        t = _text(doc)
        for bad in ("config_ridgecrest_ts.yaml", "screening_report", "config_site_a_p2.yaml",
                    "closing it kills the run", "Unpack the new release over the old folder"):
            assert bad not in t, f"{doc} still says {bad!r}"
        assert not re.search(r"^\s*(pkill|pgrep|killall)\b", t, re.M), f"{doc} tells users to kill by pattern"


# ---------------------------------------------------------------- the guide follows the code
def test_user_guide_is_where_the_help_key_looks():
    tree = ast.parse((PKG / "dashboard.py").read_text())
    names = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "GUIDE_NAMES" for t in n.targets))
    assert (PKG / names[0]).is_file(), f"the dashboard's ? key looks for {names}"


def test_user_guide_names_every_send_rule_value_and_check():
    from insar_pipeline import quality_assessment as qa
    g = _text("USER_GUIDE.md")
    flat = g.lower().replace("-", " ")
    for v in qa.SEND_VALUES:
        assert v in g, f"send rule value {v!r} missing from USER_GUIDE.md"
    for check in qa.EXPECTED_ON_FARMLAND:
        assert check.replace("_", " ") in flat, f"quality check {check!r} missing from USER_GUIDE.md"


def test_user_guide_names_every_restart_kind():
    from insar_pipeline import advice, runs
    g = _text("USER_GUIDE.md")
    for k, (label, _argv) in runs.KINDS.items():
        name = advice.KIND_WORDS.get(k, "").strip("'") or label.split(" (")[0].split(":")[0]
        assert name in g, f"run kind {k!r} ({name!r}) missing from USER_GUIDE.md"


@pytest.mark.parametrize("action,words", [("signoff", "sign off"), ("package", "package"), ("help", "guide")])
def test_user_guide_gives_the_real_key(action, words):
    src = (PKG / "dashboard.py").read_text()
    keys = re.findall(r'\(\s*"([^"]+)"\s*,\s*"' + action + r'"', src)
    if action == "help":
        keys = ["?" if k == "question_mark" else k for k in keys]
    assert keys, f"no key bound to {action}"
    g = _text("USER_GUIDE.md")
    rows = [ln for ln in g.splitlines() if any(f"`{k}`" in ln for k in keys) and words in ln.lower()]
    assert rows, f"USER_GUIDE.md does not give key {keys} for {words!r}"


def test_readme_new_settings_are_config_fields():
    # the settings table moved from README.md to TECHNICAL_REFERENCE.md (2 Nov 2026)
    from insar_pipeline.config import PipelineConfig
    fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    t = _text("TECHNICAL_REFERENCE.md")
    sec = t.split("## New settings", 1)[1].split("\n## ", 1)[0]
    named = re.findall(r"^\|\s*`([a-z0-9_]+)`\s*\|", sec, re.M)
    assert len(named) >= 15
    assert not [n for n in named if n not in fields]


def test_third_party_quotes_the_data_credits(tmp_path):
    """THIRD_PARTY.md quotes credits.py's data credit lines word for word
    (years and the CC BY link aside), so the two cannot drift apart."""
    from insar_pipeline import credits
    cfg = SimpleNamespace(project_dir=tmp_path, backend="miaplpy", start_date="2016-01-05", end_date="2026-06-17",
                          mintpy_tropo_method="pyaps", dem_path=None, dem_dir=tmp_path / "dem",
                          mine_layer_sources=(), mintpy_dir=tmp_path / "mintpy")
    rep = {"gnss": {"anchored": True}, "independent_check": {"available": True, "status": "complete"},
           "dates": ["20160105", "20260617"]}
    items = [c for c in credits.credits(cfg, rep, prov={}) if c["kind"] == "data"]
    assert {c["source"] for c in items} >= {"Copernicus Sentinel-1", "Copernicus DEM GLO-30",
                                            "Nevada Geodetic Laboratory GNSS", "OPERA DISP-S1"}

    def norm(s):
        s = s.replace(f"({credits.CC_BY_URL})", "(link)")
        s = re.sub(r"\b(19|20)\d\d(-(19|20)\d\d)?\b", "YYYY", s)
        return re.sub(r"YYYY(-YYYY)?", "YYYY", s)

    doc = norm(_text("THIRD_PARTY.md"))
    for c in items:
        assert norm(c["text"]) in doc, f"THIRD_PARTY.md does not quote the {c['source']} credit as credits.py prints it"
