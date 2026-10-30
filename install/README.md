# Installing the InSAR pipeline

Runs on Ubuntu Linux. On a Windows PC, that means Ubuntu inside WSL2, which is
free and built into Windows 10/11. The software is for use inside the operating organisation;
see `THIRD_PARTY.md` before giving it to anyone else.

**A release file is internal: contains client project configurations; do not
distribute outside the operating organisation.** It carries a client project's configs and
report areas because the test suite needs them (see "Making a release"
below); the `RELEASE.txt` inside it says the same.

## The one-step way on Windows

Put the release file, its `.sha256`, `setup.ps1` and `Setup InSAR Pipeline.cmd`
(all four come out of `make_release.sh` together) in one folder and
double-click **Setup InSAR Pipeline.cmd**. It does steps 1 and 3 below and the
memory setting, asking before anything that needs administrator rights or a
WSL restart, and finishes with an **InSAR Pipeline** icon on the desktop and in
the Start menu. Get the two accounts (step 2) first: the installer asks for
them. It is safe to run again; finished steps are skipped. Already installed
from git? `powershell -ExecutionPolicy Bypass -File install\windows\setup.ps1 -IconOnly`
(from `\\wsl.localhost\Ubuntu\home\<you>\insar_pipeline`) adds just the icon.

The rest of this page is the same thing by hand.

## 1. Windows only: get Ubuntu (once, about 10 minutes)

Open **PowerShell as administrator** and run:

    wsl --install -d Ubuntu

Restart when asked. Then open **Ubuntu** from the Start menu and choose a
username and password (they are only for this Linux system).

**Memory.** WSL gets half the PC's memory unless told otherwise. The doctor
fails below 12 GB for WSL and warns below 20 GB, so a 32 GB PC is
recommended. In Windows, create or edit
`C:\Users\<your Windows user>\.wslconfig`:

    [wsl2]
    memory=24GB
    swap=16GB

Use the PC's memory minus 6 to 8 GB (24GB on a 32 GB PC). Then run
`wsl --shutdown` in PowerShell and reopen Ubuntu. `insar doctor` (step 3)
checks this and prints the exact lines for your PC.

**Disk.** A phase-linked run peaks at about 300 GB and lives on the drive
that holds WSL's disk (usually C:). The real space available is the WSL disk's
headroom plus that drive's free space, not what Linux reports; the doctor and
the dashboard show the real figure. DISP-S1 screening needs a few GB.

## 2. Two free accounts (once)

- **NASA Earthdata**, for satellite data, OPERA products and ionosphere maps:
  https://urs.earthdata.nasa.gov/users/new. After signing up, authorise
  "Alaska Satellite Facility Data Access" under Applications → Authorized
  Apps.
- **Copernicus Climate Data Store**, for ERA5 weather (projects outside the
  continental US, or before July 2016; US projects use NOAA HRRR, which needs
  no account):
  https://cds.climate.copernicus.eu. After signing in, open the ERA5 hourly
  data on pressure levels dataset, go to Download data → Terms of use and
  accept the **CC-BY licence**. Your API token is on your profile page.

## 3. Install (about 30 minutes, mostly downloading)

**From the GitHub repository** (Kyle adds you as a collaborator). Make the
SSH key in the Ubuntu window, not in PowerShell: git in Ubuntu cannot use a
key made on the Windows side. Add the printed key on github.com under
Settings, SSH and GPG keys, then clone and install:

    ssh-keygen -t ed25519
    cat ~/.ssh/id_ed25519.pub
    ssh -T git@github.com
    git clone git@github.com:Kyle-Wire/insar_pipeline.git ~/insar_pipeline
    bash ~/insar_pipeline/install/install.sh

To update later: `cd ~/insar_pipeline && git pull`, then re-run the installer
(finished steps are skipped).

**From a release file.** Copy the release file,
`insar_pipeline_<version>_<code>.tar.gz`, and the `.sha256` file beside it
into your Ubuntu home folder, and check it arrived intact:
`sha256sum -c insar_pipeline_*.tar.gz.sha256` (it prints OK). In Windows Explorer that is
`\\wsl.localhost\Ubuntu\home\<your Ubuntu username>`. Then, in the Ubuntu
window:

    cd ~
    tar -xzf insar_pipeline_*.tar.gz
    bash insar_pipeline/install/install.sh

The installer:
- asks for the Earthdata login, the CDS token and a projects folder
  (Enter accepts `~/insar_projects`, or on a re-run the folder already in
  use);
- installs conda and four Python environments at the exact versions this was
  validated with (`insar`, `miaplpy`, `burst2safe` for the downloads, and
  `raider-env` for the NOAA HRRR weather delays),
  snaphu 2.0.7 built without its noncommercial CS2 module, and the patched
  MiaplPy (see `patches/miaplpy/README.md`);
- puts the `insar` command in `~/.local/bin`;
- finishes with `insar doctor`, which should end in **Ready.**

It is safe to re-run; finished steps are skipped.

Then tell the doctor who prepares the reports (printed in every report's
footer, and sent to OpenStreetMap as the contact for place searches):

    insar doctor --firm "Your Firm" --preparer "Your Name" --contact "you@yourfirm.com"

## 4. Use it

Double-click **InSAR Pipeline** on the desktop: the app window (projects, live
runs on a map, results). Or open a new Ubuntu window and type:

    insar

That opens the terminal dashboard (`insar app` opens the window from there). Press `?` in it for the user guide
(`USER_GUIDE.md`): the quality checks, when a report can be sent, what every
number means, and what to do when a run fails. The full run, step by step,
is `INSAR_RUNBOOK.md`.

- **Projects screen:** Enter opens a project, `b` opens the map builder
  (draw or upload the area and the mine outlines; it creates the project), `n`
  starts a new area inside the terminal instead, `v` opens Results (any area,
  any dates), `f` browses files, `?` the guide, `q` quits.
- **A project:** `s` starts a run (the recommended one first, after the
  preflight checks), `x` stops it, `p` preflight, `e` edits the config, `o`
  opens the client report, `r` signs it off, `k` builds the client package.

Runs keep going if you close the window.

Other commands:

    insar builder          # the map builder: draw/upload the area and mine outlines, create the project
    insar doctor           # check the installation
    insar results <config.yaml> --area "Area A2" --from 2020-01-01 --to 2024-12-31 --export
                           # any area and date range: per-point rates, yearly rates, statistics, Excel
    insar cleanup <config.yaml> --level 1     # what can be deleted from a finished project (dry run)
    insar help

**Routes.**
- **Phase-linked:** official results. 1 to 2 days and about 300 GB per area.
- **OPERA DISP-S1:** fast screening. A few hours; North America only. Not
  for official numbers, and its reports never pass the release gate.

## 5. What leaves this computer

| Service | Receives |
|---|---|
| ASF (NASA) | the area and dates (archive search); the processing box per track (burst catalogue); the Earthdata login |
| Copernicus CDS (ERA5) | the processing box and the dates |
| Nevada Geodetic Laboratory | GNSS station IDs |
| OpenStreetMap Nominatim | the builder's search text, and the area's centre for the report's place name |
| OpenStreetMap Overpass | the processing box, once per project, for the water map used to check map positions |
| Map tile servers (USGS by default; Esri, OpenStreetMap or OpenTopoMap if chosen) | whatever area you view in the builder |
| cdnjs, Google Fonts | the builder page's map library and fonts |

Type a town or road in the builder's search, never a client's or a site's
name. The full list, with when each service is used, how it authenticates and
its approval status, is `EXTERNAL_SERVICES.md`.

## Updating to a new release

Move the old folder aside (it is your way back), unpack the new release in a
fresh `insar_pipeline` folder, and run the installer from it:

    cd ~
    mv insar_pipeline "insar_pipeline_old_$(date +%Y%m%d)"
    tar -xzf insar_pipeline_<version>_<code>.tar.gz
    bash insar_pipeline/install/install.sh --no-credentials
    insar --version

Unpacking over the old folder is not safe: files removed or renamed in the
new release would stay behind and keep running.

The installer repoints the `insar` command and the Python install at the new
folder, and moves the patched MiaplPy's pip lock to it. Your projects,
settings (`~/.config/insar/settings.json`) and logins are not touched: the
projects folder stays the folder already in use (`~/insar_projects`, or the
one set with `insar doctor --projects`).

It keeps each existing environment, but checks it against the new release's
pins in `install/envs/`. When one differs, it lists the differences and the
command that rebuilds it, for example:

    conda env remove -n insar && bash ~/insar_pipeline/install/install.sh

A snaphu 2.0.7 that still contains the CS2 module is rebuilt without it; the
official route's unwrapping is unchanged by this (checked bit for bit, see
`THIRD_PARTY.md`).

To go back: `mv insar_pipeline insar_pipeline_bad && mv insar_pipeline_old_<date> insar_pipeline`,
then run `bash insar_pipeline/install/install.sh --no-credentials` again.
Delete the old folder once the new release has run a project.

## Making a release (Kyle)

Bump `version` in `pyproject.toml`, run the tests, then:

    bash ~/insar_pipeline/install/make_release.sh [output folder]

It writes `insar_pipeline_<version>_<code hash>.tar.gz` (default: your home
folder) from a whitelist: the code, `scripts/`, `install/`, the MiaplPy patch,
`tests/`, `context/`, the documents, and a `RELEASE.txt`
naming the build. It never ships `.claude/`, Windows `:Zone.Identifier`
files, the stray root `config.yaml` and `project_zone.geojson`, `backups/`,
`review_*/`, the archived configs in `configs/archive/` (only the archived
topsApp runbook ships), `report_assets/`, caches, or anything from the
projects folder, and refuses to build if any of those slips in.

It can carry client work, on purpose. Built on the developer's machine, it
takes the machine's local-only tests (`tests/test_local_*.py`, not in the git
repository) and the client report areas in `context/`, because those tests
load them; `context/` also holds public Indiana mine layers (IGWS and Indiana
DNR; `context/SOURCE.txt`). No project config ships (28 Oct 2026): projects
keep theirs in the projects folder's `configs/`. So every release is internal: contains client
project configurations; do not distribute outside the operating organisation. `RELEASE.txt`
carries that line, and the build prints it. The engineering log (HANDOFF.md)
does not ship: it stays on the developer's machine.

## Notes

- The three comparisons against `opera-utils` in the test suite show as
  SKIPPED (`pytest -rs` lists them): `opera-utils` is not installed in the
  `insar` environment.
- snaphu is built with `-DNO_CS2`. A config asking for `INITMETHOD MCF`
  stops with "program not compiled with cs2 MCF solver module". The official
  route uses MST. The classic route's `isce` unwrap engine (comparison only)
  uses the snaphu built into ISCE2, which contains CS2; keep
  `stack_unw_full_cost: true` there.
- To rebuild snaphu by hand: `bash install/build_snaphu.sh miaplpy`
  (`SNAPHU_TARBALL=<file>` uses a local copy of the source, `SNAPHU_DEST=<dir>`
  installs elsewhere for a test).
