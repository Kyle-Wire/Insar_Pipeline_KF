# Changing insar_pipeline

For whoever changes the code. To run the pipeline, start with
[README.md](README.md) and [USER_GUIDE.md](USER_GUIDE.md).

## Set-up

1. Install the pipeline normally (`install/README.md`). For development,
   clone the repository to `~/insar_pipeline` and run its installer. The
   folder is the Python package `insar_pipeline`, imported from your home
   folder.
2. Turn on the client-data check for every commit, once per clone:

        git config core.hooksPath .githooks

3. Work in the `insar` environment, from your home folder:

        conda activate insar && cd ~

## The client-data rule

Client data never goes in the repository: project configs, site outlines,
coordinates, report numbers, client or site names, project folders and
products. They stay on the operator's machine. `.gitignore` keeps the known
files out, and `scripts/check_repo_clean.sh` refuses a commit that tracks
one.

- In code, tests and docs, call a site "the example site" or use a public
  validation site (`configs/experiments/README.md` records them; their configs
  left the repository on 28 Oct 2026). A test writes the small config it needs
  in its own temporary folder. The first client projects appear as Site A to
  Site D and their report areas as Area A1 to A3.
- A test that has to read a real client project, config or delivered record
  goes in `tests/test_local_<name>.py`. Those files are ignored by git and
  run only where the client files are.
- To have the check also catch names, put the names and coordinates of your
  client projects, one per line, in `~/.config/insar/client_markers.txt`.
  This file never goes in the repository. The pre-commit hook then refuses
  any added line that contains one.
- Never paste client details into an issue or pull request either.

## Tests

The full suite takes about 9 minutes:

    python -m pytest insar_pipeline/tests -q -p no:cacheprovider

- **Each fix ships with a test that fails on the old code.** The test's
  docstring says what it guards and the date.
- **Tests run on synthetic projects in `tmp_path`.** They use no network
  and no real project. `tests/conftest.py` keeps them away from the real
  notifications log and the real run registry.
- **A new optional block in the client report** (`client_report._optional`)
  needs a key in `OPTIONAL_BLOCKS`. A block made while the figures or the
  page are drawn must stay out of `release.OMITTED_HARD` and inside
  `release._OMITTED_NOT_HASHED`.
- **Tests that need a client project or a local record skip** on a fresh
  clone, and run on the machine that has them. The golden test pins the
  client numbers of real projects; it runs only when asked:

      INSAR_GOLDEN=1 python -m pytest insar_pipeline/tests/test_fix_INT.py

- **pyflakes must stay clean.** A test enforces it, and the GitHub checks
  run it on every push.
- **Changes that move client numbers:** preview the report into a scratch
  folder (`--out-dir`) before you overwrite a delivered one. Then write down
  which projects move and by how much.

## Where things are

The package folder is flat: one module per job. Each module's docstring
says what it does and why.

| Stage | Modules |
|---|---|
| Entry points | `dashboard` (the `insar` command), `pipeline` (all stages in order), `cli` (old menu), `doctor` (is this machine ready), `runs` (detached runs, project index, run-log level and size cap), `progress`, `procs`, `notify`, `advice` (plain cause and next step after a stop), `diagnose` (`insar diagnose`: the custodian's zip), `integrity` (installed files against the release manifest) |
| The app window | `app_server` + `app_assets/` (`insar app`: projects, live runs on a map, results, sign-off), `file_manager` (the Files page: sizes and deletions, logged), `map_export` (printable map pages, PDF / PNG) |
| Setting up a project | `map_builder` + `project_builder.html` (draw or upload the area and mine outlines), `areas` (outline to project, production defaults in `write_project`), `config` (load and validate), `settings` (machine paths), `plan_run` (size and cost) |
| Data and processing | `data_acquisition`, `pair_selection`, `stack_backend` (ISCE2 topsStack), `miaplpy_backend` (phase linking, the official route), `unwrap_crop`, `tropo_correction`, `weather_raider` (NOAA HRRR delays through RAiDER, the default weather in the continental US), `corrections` (tides, ionosphere), `time_series` (MintPy), `disp_backend`/`disp_chain`/`tropo_disp`/`static_layers` (OPERA DISP-S1 screening), `nisar_backend`/`lband` (L-band, not validated), `nisar_gslc`/`nisar_dolphin`/`nisar_pl` (the phase-linked NISAR route, comparison only), `nisar_mining` (an experiment, not wired in), `hyp3_backend`, `isce_processing`, `isceenv` |
| Quality and measurement | `quality_assessment` (the QA and its send rule), `stack_qc`, `significance` (how far a figure can be trusted), `gnss_tie`, `gnss_validation`, `plate_motion`, `landcover`, `aliasing` (fast motion), `resolution`, `crosscheck` (against DISP-S1), `injection_test` |
| Products and the client | `deliverables` (standard products), `gis_products`, `figures`, `field_points`, `client_report` + `report_insights` (the report and what it says), `client_kmz`, `credits`, `provenance`, `version` (release + code hash), `release` (the send gate and sign-off), `package` (what leaves the building) |
| Looking at results | `results`, `results_screen`, `results_viewer` + `results_viewer.html` + `viewer_assets/` (the results map), `viewer_bundle` (the Python reference for the map's browser arithmetic), `files_screen`, `events`, `panel_history`, `decompose`, `combine` + `combine_viewer` (an ascending and a descending project into vertical and east-west) |
| Screening layers (not client figures) | `local_change` (the local change layer), `feature_check` (long-pair feature check), `vendor_points` + `scripts/compare_vendor_points.py` (like-for-like with a vendor's point export) |
| Housekeeping and validation | `storage`, `archive_project`, `scripts/cleanup.py`, `ground_truth`, `validation` (the harness for `configs/experiments/`), `scripts/make_sbom.py` (the release's SBOM) |

Configuration lives in `config.py`, one dataclass field per setting, with
the basis of its default beside it. New projects take the production
defaults from `areas.write_project`. Machine paths live only in
`settings.py`. Each project's state lives in its own folder: `run.json`,
`run_result.json`, `progress.jsonl`, `logs/` and `deliverables/`.

**A common change, a new setting:**
1. Add the field to `PipelineConfig` in `config.py`, with its validation and
   a comment giving the basis of the default.
2. If new projects should get it, add the default in `areas.write_project`.
3. Add a test that fails without the change.
4. Update the doc that describes the behaviour.
5. Add a CHANGELOG entry.

## Lint

Every push runs pyflakes and ruff (`checks.yml`; the rules are in
`pyproject.toml`, `[tool.ruff]`): pyflakes' checks, syntax errors, the
import, lambda and statement rules ruff calls E4 and E7, and blind
`except Exception` (BLE001). Not enforced, on purpose, as house style:
several short statements on one line with `;` (E701, E702) and one-letter
names in the maths (E741).

An accepted exception is a `# noqa: <code>` on the line, with the reason
beside it where it isn't obvious. They are accepted in these cases only:

- **BLE001** (catch everything): where a failure must not stop the larger
  job and is recorded instead: a report part listed under "Left out of this
  build", a credit or provenance field written as unknown, a doctor check
  shown as failed, a cleanup that carries on. Never around a figure that goes
  into a report.
- **E402** (import not at the top): scripts that put the package on the path
  first, and imports that need an environment variable set before them.
- **E731** (lambda assigned to a name): short local key or formatting
  functions.
- **F401 / F841**: names kept for older readers.

Run `ruff check .` and `python -m pyflakes *.py scripts/*.py tests/*.py`
before you push.

## Commits, branches, releases

- `main` is what coworkers install. Make changes on a branch and merge
  them through a pull request. The pull-request template is the checklist.
- **Release:**
  1. Bump `version` in `pyproject.toml`.
  2. Add a [CHANGELOG.md](CHANGELOG.md) entry.
  3. Run the full suite and commit.
  4. Tag the release and push the tag:

         git tag v<version> && git push && git push origin v<version>

     The tag runs `.github/workflows/release.yml`: it refuses a tag that is not
     `v` + `pyproject.toml`'s version, runs the checks and the full suite, and
     then publishes a GitHub Release with the tarball, its `.sha256` and the
     Windows setup files. The Release is as private as the repository.
  5. `install/make_release.sh` builds the same tarball by hand, for a machine
     without repository access. RELEASE.txt inside it names the version and
     code hash.
- **GitHub checks:** every push runs `checks.yml` (client-data check,
  pyflakes, the viewer's scripts, about a minute). Pull requests, `main` and
  each release also run `tests.yml`, the full suite in the pinned `insar`
  environment (Actions > tests > Run workflow runs it on any branch). Tests
  that need a client project skip there, as on any fresh clone.
- **Problems:** GitHub issues (template: "Problem report"). Kyle Wire reads
  them.
