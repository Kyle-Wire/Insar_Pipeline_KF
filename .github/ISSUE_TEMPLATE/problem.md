---
name: Problem report
about: A run that failed, a wrong-looking number, a screen that misbehaves
title: ""
labels: bug
---

<!-- Do NOT paste client names, site coordinates, report numbers or report
pages here. Refer to the project by its folder name. Attach files only from
the list below, after checking they hold nothing client-identifying. -->

**What happened, and what you expected**


**Project folder name and stage** (e.g. `example_site`, MintPy / Products / sign-off)


**Build** (`insar --version`, e.g. `insar_pipeline 2026.10.16 (code 1a2b3c4d5e6f)`)


**What the run left behind** (in the project folder)
- `run_result.json`: the `plain_cause`, `next_action` and `technical` lines
- the last 40 lines of the newest log in `logs/`
- `insar doctor` output, if the machine may be the cause
