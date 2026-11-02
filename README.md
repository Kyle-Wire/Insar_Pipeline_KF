# InSAR Ground Movement Pipeline

A desktop program that measures how the ground at a site has moved — sinking
or rising, in inches or millimetres per year — using free satellite radar
images, and turns the result into a draft client report, a Google Earth file
and GIS data for a qualified reviewer to check and sign off.

It is built for the Geotechnical group, for screening sites for long-term
ground movement such as subsidence over underground mines, groundwater
pumping or soft soils. Maintainer: Kyle Wire (Geotechnical).

**Internal use only.** The software and its documentation are licensed for use
inside the operating organisation (see [LICENSE](LICENSE)) and must not be
shared outside it. The reports and data it produces may go to clients once
signed off.

---

## What it does

1. **You outline the site.** In the map builder you draw or upload the area of
   interest, and optionally mine outlines and the report areas a client cares
   about. The program checks which satellite images exist.
2. **It downloads the images.** Copernicus Sentinel-1 radar images of the site,
   taken every 12 days, from 2016 to the present, from NASA's free archive.
3. **It measures the movement.** It compares the images with each other to
   find millimetre-scale changes in the distance from the satellite to the
   ground, corrects them for the weather and the terrain, and builds a
   movement history for every 15 m square of ground.
4. **It judges what is real.** Each figure is compared with the ordinary
   ground around the site, tied to nearby GNSS (GPS) stations, given a 95%
   range, and run through quality checks that say whether the report can be
   sent, needs a specialist, or must not be sent.
5. **It writes the products.** A client report (HTML and PDF), a Google Earth
   (KMZ) file, GeoTIFF maps, a GeoPackage of measurement points and
   spreadsheets.
6. **A person signs it off.** Every report stays marked DRAFT until a named
   reviewer signs it off. Only then can the client package be built.

A full run takes about a day of computer time, most of it unattended. You can
close the program while it runs; the run carries on.

## What it is good for, and what it is not

| Good for | Not for |
|---|---|
| Slow to moderate subsidence (a few mm/yr to tens of cm over years), with a stated uncertainty | Fast longwall mining subsidence from Sentinel-1: the satellite cannot keep up |
| Multi-year rates and total movement since 2016 | Single-year rates to better than about 12 mm/yr |
| Screening: where is the ground moving, by about how much, compared with its surroundings | A survey: it does not replace levelling, GNSS monitoring or inspection |
| Areas with open ground, farmland and built-up land | Dense woods (removed from the products) or water |

What has been checked against independent GPS measurements, and how well it
did, is in [VALIDATION.md](VALIDATION.md). Everything the program cannot yet
do, in plain words, is in [LIMITATIONS.md](LIMITATIONS.md). Read both before
quoting a figure to a client.

## What you need

- A Windows 10 or 11 PC with **32 GB of memory** (at least 24 GB; a 16 GB PC
  does not pass the installation check).
- **Free disk space:** a full ten-year project peaks near 300 GB while it runs,
  and the installation check asks for 350 GB free. About 180 GB stays
  afterwards, and most of that is freed once the report is final. A shorter
  record or a smaller area needs less.
- **Administrator rights once**, to switch on Ubuntu inside Windows (WSL). Ask
  IT if you don't have them.
- An internet connection, and two free data accounts (NASA Earthdata and the
  Copernicus Climate Data Store). The install guide walks you through both.
- The release files, from the maintainer.

## Installing

Follow [install/README.md](install/README.md). On Windows it is one
double-click: the setup switches on Ubuntu, installs the program and its
tools, checks the installation and puts an **InSAR Pipeline** icon on your
desktop. Allow one to two hours, mostly downloading and the final check; you
need to be there only at the start, after a restart, and when it asks for your
accounts.

## Using it

Double-click **InSAR Pipeline** on the desktop. The program opens in its own
window, with your projects on a map. From there you build a project, start the
run, watch its progress, read the report, explore the results map, export map
pages, sign off and build the client package.

[USER_GUIDE.md](USER_GUIDE.md) starts with a ten-step first project and
explains every screen, every number in the report, the quality checks and
what to do when something stops. The same guide opens from the program (press
`?` in the dashboard).

## Getting help

- `insar doctor` checks the installation and says how to fix anything missing.
- `insar diagnose <project>` makes one zip of the run records and logs to send
  to the maintainer when something goes wrong. It contains no client outlines,
  report, maps or point files.
- The troubleshooting chapter of the [user guide](USER_GUIDE.md).
- The maintainer, Kyle Wire. Refer to a project by its folder name, never by
  the client's name or coordinates.

## Documents

| Read | For |
|---|---|
| [install/README.md](install/README.md) | installing on a new PC, updating, and what to do if the install stops |
| [USER_GUIDE.md](USER_GUIDE.md) | using the program: your first project, every screen, reading the results, sign-off, troubleshooting |
| [LIMITATIONS.md](LIMITATIONS.md) | what the program cannot yet do, its settings and the reasons behind them |
| [VALIDATION.md](VALIDATION.md) | what it has been tested against, and where it fails |
| [INSAR_RUNBOOK.md](INSAR_RUNBOOK.md) | a run step by step, from install to cleanup, for the operator |
| [PRIVACY.md](PRIVACY.md) | the personal and client information the program keeps, and how to remove it |
| [EXTERNAL_SERVICES.md](EXTERNAL_SERVICES.md) | every outside service it contacts, and what it sends |
| [THIRD_PARTY.md](THIRD_PARTY.md) | the licences of the software and data it uses, and the credits every client output must carry |
| [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md) | how it works inside: commands, processing routes, every output file, settings, measurement and design |
| [CHANGELOG.md](CHANGELOG.md) | what changed, release by release |
| [CONTRIBUTING.md](CONTRIBUTING.md) | changing the code: set-up, tests, lint, the client-data rule, releases |

## For the maintainer

**Status.** In use for slow and moderate subsidence with Sentinel-1 (C-band)
on the phase-linked route; every client report is a DRAFT until a named
reviewer signs it off. Not validated, and not for client numbers: fast
longwall subsidence from C-band (it fails, see VALIDATION.md) and L-band
NISAR results (unverified). The validation used ERA5 weather and no wooded
mask, and today's defaults differ: see [LIMITATIONS.md](LIMITATIONS.md).

**This repository holds no client data, ever.** Client project configs, site
outlines, report numbers and project folders live on the operator's machine,
under the projects folder (`insar doctor` shows where). `.gitignore` keeps the
known ones out, and `scripts/check_repo_clean.sh` (run by the automated checks
and by the pre-commit hook in `.githooks/`) fails if one is about to be
committed. Tests that need a client project skip on a fresh copy.

**Versions and releases.** The version is `pyproject.toml`'s; each release
bumps it, adds a [CHANGELOG.md](CHANGELOG.md) entry and is tagged
`v<version>`. Every output records the version and the 12-character code hash
of the build that made it (`insar --version`). `install/make_release.sh`
builds the internal release files; CONTRIBUTING.md has the steps.

**Requirements.** The environments are pinned exactly in `install/envs/`
(`insar.yml`, `miaplpy.yml`, `burst2safe.yml`, `raider-env.yml`,
`dolphin.yml`; conda-forge). `requirements.txt` is a reference list of the
packages the code imports, at those versions, not the install source; every
release carries `SBOM.cdx.json`, the full list. The pipeline is proprietary;
see [LICENSE](LICENSE). Third-party licences and required credits:
[THIRD_PARTY.md](THIRD_PARTY.md).
