# InSAR pipeline: user guide

This guide is for the people who run projects with the InSAR pipeline and
deliver its results. It assumes no specialist knowledge of radar
interferometry. It explains the terms you will meet, the process from a new
project to a delivered report, how to read the results, and what to do when
something goes wrong. The dashboard opens this guide when you press `?`.

**New to the program?** Read "Start here", just below. It needs no knowledge
of radar, Linux or programming, and it walks you through a first project in
ten steps. The numbered chapters after it hold the full detail.

Related documents:

- `install/README.md`: installing the software on a new machine.
- `INSAR_RUNBOOK.md`: the operator's checklist for a production run, with
  measured run times and the technical detail of each stage.
- `README.md`: the technical description of the pipeline and its settings.
- `PRIVACY.md`: the personal and client information the program keeps, and
  where.

**Contents**

- **Start here** (for a first-time user)
  - What the program does
  - What it is for, and what it is not
  - Before you start: the two windows, and typing a command
  - Your first project in 10 steps
  - How long things take and how much space they need
  - Words you'll see
  - Where to get help
1. What the pipeline does: the method in brief
2. Key terms: every word the program uses, in plain language
3. The process at a glance: the stages of a run
4. Running a project, step by step: the detail behind the ten steps
5. The dashboard: every screen, button and key
   - 5.1 The app window
   - 5.2 The terminal dashboard
6. Interpreting the results: reading the report's figures
   - 6.7 The results map
   - 6.8 The Google Earth file (KMZ)
   - 6.9 Common misreadings
7. Quality checks, the send rule and release: whether a report can go to a
   client
   - 7.4 Tracing a reported figure back to its source
8. Troubleshooting: when a run stops or something does not work
   - 8.5 Sending a problem to the custodian (`insar diagnose`)
   - 8.6 Run logs and their level
9. Disk space and memory
   - 9.1 Where the program writes
   - 9.2 Removing a client's data
10. Settings you may need to change
11. Getting help

---

## Start here

This chapter is for your first days with the program. It assumes you have
never used Ubuntu or typed a command. Read it once from top to bottom. Each
step names the section later in this guide that has the full detail.

### What the program does

The program measures slow movement of the ground surface from satellite
radar images. The Copernicus Sentinel-1 satellites have imaged the same
ground about every 12 days since 2016. By comparing those images, the program
works out how much the ground at a site has moved: a vertical rate in inches
per year, with millimetres per year beside it. Each figure is compared with
ordinary ground around the site, measured in the same images, and tied to
permanent GNSS (GPS) stations nearby, which show how the whole region is
moving. The program then writes a draft client report, a Google Earth file
(KMZ) and GIS data (GeoTIFF maps and measurement points). Nothing goes to a
client until a qualified reviewer has checked the report and signed it off.

### What it is for, and what it is not

Good for:

- Slow ground movement over several years, such as subsidence of a few
  millimetres a year over old mine workings. This is what passed its check
  against independent GNSS stations (`VALIDATION.md`).
- Rates over several years, and the total movement since the record began,
  each with a stated 95% range.
- Screening: finding where the ground moves differently from its
  surroundings, and by about how much.
- Open ground, farmland and built-up land.

Not for:

- Fast subsidence over active longwall panels, from Sentinel-1. The
  satellite cannot keep up: in a test it caught only about 40% of the drop.
  The program does not stop you; the reviewer must apply this rule at
  sign-off.
- Rates for a single year to better than about 12 mm/yr (about half an inch
  a year).
- Ground under dense woods (left out of the maps by default) or water (usually
  not measured).
- Results from the NISAR satellite. They are labelled "Comparison - NISAR,
  not validated" and are for comparison only, never a client finding.
- The DISP-S1 screening route and the local change layer (section 6.10).
  They show where to look; they are never released to a client.
- Replacing a survey. The report is a screening of ground movement, not a
  survey: it does not replace levelling, GNSS monitoring or inspection.

Everything the program cannot yet do is listed in `LIMITATIONS.md`.

### Before you start: the two windows, and typing a command

The program is installed once, usually by the maintainer (see
`install/README.md`). After that you use two windows:

- **The app window.** Double-click **InSAR Pipeline** on the Windows desktop
  (it is in the Start menu too). Most of the work is done here, with buttons.
  Its pages are listed down the left side: **Projects**, **New project**,
  **Files** and **This computer**.
- **The Ubuntu window** (also called the terminal). Use it for the few things
  the app cannot do: checking the computer (`insar doctor`), editing a
  project's settings, and the other commands in this guide. To open it,
  click the Windows Start menu, type
  `Ubuntu` and click the **Ubuntu** app. A black window opens with a line of
  text that ends in `$`.

**Typing a command.** Wherever this guide shows a command in a box like this
one, type it in the Ubuntu window exactly as shown, then press Enter:

    insar doctor

Replace any part in angle brackets with your own value, and leave out the
brackets. For example, `insar diagnose <project>` for a project called
Site B becomes:

    insar diagnose site_b

(A project's folder name is its name in lower case, with `_` in place of
spaces.) Typing `insar` on its own opens the terminal dashboard, where you
work with single keys (section 5.2). Press `q` to leave it. Closing the
Ubuntu window, the app window or the dashboard never stops a run.

### Your first project in 10 steps

Each step says where to do it: in the **app** window, or in the **Ubuntu**
window.

1. **Check the computer** (once, and after any change to the computer).
   *Ubuntu:* type

       insar doctor

   It checks the logins, the disk, the memory and the software. It must end
   with **Ready.** Each red line says how to fix it. Then, in the *app*, open
   **This computer**, fill in **Firm**, **Prepared by** and **Contact** under
   **Company**, and press **Save**. Every report prints these, and a report
   cannot be released without the firm and the preparer. (Section 4.1.)
2. **Gather the mine outlines.** Ask the client or the mine operator for
   current mine plans, or use state records. Accepted files: a zipped
   shapefile, KML or KMZ, GeoJSON, or a GeoPackage. Note where each came from
   and how current it is. Missing mine outlines are the most common cause of
   misleading results. (Section 4.2.)
3. **Build the project.** *App:* press **New project**, then **Open the map
   builder**. The builder opens in your web browser. Work down its numbered
   steps:
   - **1 Project area:** press **Draw rectangle** or **Draw polygon**, or
     **Upload** the outline file.
   - **2 Mines and known movement:** upload each layer under its heading
     (Underground workings, Surface mines, Other known movement, Dated mined
     areas) and fill in its **Source, date** box.
   - **3 Extra report areas** and **4 Reference point** are optional. Leave
     the reference point empty unless you know a stable spot.
   - **5 Dates and route:** keep the dates offered and choose
     **Phase-linked**.
   - **6 Check the satellite archive:** press **Check archive and GNSS**. It
     warns when fewer than 3 GNSS stations are close enough.
   - **7 Create the project:** type the plain name the client uses for the
     site, then press **Create project**.

   The project appears on the app's **Projects** page within seconds.
   (Section 4.3.)
4. **Check and start the run.** *App:* on **Projects**, click the project's
   card, then press **Start**. Choose the first option, marked
   **Recommended** (for a new project, Whole pipeline). The checks run, which
   can take a minute. When none fails, the label reads **ready**: press
   **Start the run**. A line marked XX must be fixed first; it says how. Then
   press **Check again**. (Section 4.4.)
5. **Let it run.** A whole run takes 1 to 2 days, most of it unattended. The
   project's page shows the six steps of the run, a card with the time left,
   and four tabs: **Stages**, **Results**, **Log** and **Activity**. You can
   close the app window: the run carries on. Only **Stop** stops a run. Keep
   the computer on: a Windows restart interrupts the run (it can then be
   resumed, section 8.4). A Windows notification appears when the run ends.
   (Sections 4.5 and 5.1.)
6. **Read the verdict and the report.** *App:* the **Results** tab shows
   DRAFT or FINAL, the send rule ("send", "send with notes" or "hold for
   specialist") and what is still to settle. Press **Report** to open the
   client report. While it is a DRAFT the app asks first; press **Open the
   draft**. Read it the way the client will. If the send rule is "hold for
   specialist", refer the project to the specialist before anyone signs
   (section 7.2). (Sections 4.6, 6 and 7.)
7. **Check every feature and explore.** Check every feature the report lists
   against aerial imagery and dates: spoil, fill or a new pit can look like
   subsidence. *App:* **Results map** opens a map of every measured point in
   your browser, where you can pick dates and draw areas. Its short-window
   figures are a look, not figures to quote. (Sections 4.6, 6.4 and 6.7.)
8. **Sign off.** Only after reading the whole report and checking every
   feature. *App:* press **Sign off** (beside **Report**). The dialog lists
   first anything that still blocks a final release whatever you sign. Type
   your name under **Reviewer**, tick each feature, add the specialist's name
   for a held report and any **Notes**, then press **Sign off, rebuild the
   report and package**. The sign-off is recorded at once. The report rebuild
   and the client package take a few minutes more, and the **Results** tab
   says how it ended. If the rebuild changed a number, sign again.
   (Section 4.7.)
9. **Send the client package.** When the report is FINAL, the sign-off builds
   the package in the project folder, under
   `deliverables/releases/<YYYYmmdd_HHMM>/`. To find it: *app* **Files**, pick
   the project, press **Open in Explorer**. Send the `client/` folder (with
   `README.txt` and `MANIFEST.json`), and `data/` only if the client wants GIS
   files. Never send `technical/`, and never send files straight from
   `deliverables/`. If no package was built, open the terminal dashboard
   (*Ubuntu:* type `insar`), open the project with Enter and press `k`.
   (Section 4.8.)
10. **Free the disk.** Once the FINAL package is built, the processing files
    are deleted automatically and the results kept. To do it by hand: *app*
    **Files**, pick the project, press **Keep results only**. It shows what
    it will delete and asks first. Deleting is permanent: Ubuntu has no
    recycle bin. (Sections 4.9 and 9.)

### How long things take and how much space they need

| Task | Time | Disk space |
|---|---|---|
| Switching on Ubuntu in Windows (once) | about 10 minutes | |
| Installing the program (once) | about 30 minutes, mostly downloading | several GB for the software |
| The checks before a run (preflight) | about a minute | needs about 350 GB free for a phase-linked run (the checks warn below that) |
| A whole phase-linked run | 1 to 2 days | peaks at about 300 GB |
| ... of which: downloading the images | about 4.5 hours (Site A, 278 dates) | about 0.45 GB per date |
| ... lining up the images (coregistration) | about 2 hours (Site A) | |
| ... phase linking | about 21 hours (Site A), the longest stage | |
| A finished project, before cleanup | | about 180 GB |
| Rebuilding the report and package after sign-off | a few minutes (the client report itself took about 2.5 minutes at Site A) | little |
| After Keep results only (automatic once the FINAL package is built) | | most of it is freed; the report and deliverables (about 2 GB), the logs and the final time-series products stay |
| A DISP-S1 screening run | a few hours | a few GB |

**Memory.** A computer with 32 GB of memory is recommended. `insar doctor`
fails below 12 GB for Ubuntu and warns below 20 GB (section 9).

**Weather data.** The ERA5 weather data used for the atmospheric correction
outside the continental US are published about 5 days late, so the map
builder sets the end date to a week before today.

### Words you'll see

Section 2, "Key terms", explains every word the program uses, in plain
language. The ones you will meet first are: InSAR, Sentinel-1, line of
sight, vertical, coherence, GNSS and the GNSS tie, comparison ground
(ordinary ground), report area, feature, DRAFT and FINAL, sign-off, hard
blocker, send rule, phase-linked route, DISP-S1 screening, project folder,
terminal, and Ubuntu and WSL.

### Where to get help

- **`insar doctor`** (in the Ubuntu window) checks the installation and says
  how to fix anything missing.
- **When a run stops**, open the project: it gives the plain cause and a
  **What now:** line. Section 8 says what to do for each stage.
- **`insar diagnose <project>`** makes one zip of the project's run records
  and logs to send to the maintainer (section 8.5). It holds no client
  outlines, report, maps or point files.
- **The maintainer, Kyle Wire** (the custodian and specialist this guide
  refers to). Section 11 says what to include. Refer to a project by its
  folder name, never by the client's name or coordinates.

---

## 1. What the pipeline does

The pipeline measures slow movement of the ground surface, such as
subsidence over mine workings, from satellite radar images. It uses the
Sentinel-1 satellites, which image the same ground about every 12 days, and
turns roughly ten years of those images into a rate of vertical movement, in
inches per year, for the areas you name.

Three principles shape every number it reports:

- **Every figure is relative to ordinary ground.** Each area is compared with
  ordinary ground around the site (farmland, by default), measured in the
  same images over the same dates. Errors common to the whole scene cancel
  out in that comparison.
- **Figures are tied to the regional movement.** Permanent GNSS (GPS)
  stations nearby measure how the region as a whole is moving. That regional
  rate is added to the relative figures, so the totals describe real
  movement, not only movement relative to the surroundings.
- **Chance is ruled out before movement is claimed.** Ordinary ground never
  reads exactly zero. A figure that ordinary ground could produce by chance
  is reported as "within normal ground variation", not as movement.

### Processing routes

The pipeline can process a project in three ways. Only one produces official
results.

| Route | Config `backend` | Use | Typical cost |
|---|---|---|---|
| Phase-linked | `miaplpy` | **Official results.** The only route whose report can be released to a client. | 1 to 2 days and about 300 GB of disk per project |
| OPERA DISP-S1 | `disp` | Screening only: a quick look at where movement might be. No client report unless `disp_client_report: true`, and the release gate never releases it. | a few hours |
| Classic stack | `stack` | Comparison with the phase-linked route only. | — |

Unless you have a specific reason, use the phase-linked route.

---

## 2. Key terms

These terms appear in the dashboard, the report and the rest of this guide.
Look a word up here whenever you meet it; you do not need to learn them in
advance. They are in alphabetical order.

**Area, report area.** A named outline the report gives figures for. The
project area (the outline the client drew) is always one; extra report areas
and mine layers marked "report its numbers" add rows. Names must be the plain
names the client uses.

**Coherence.** How consistently the radar signal from a patch of ground
repeats between images, from 0 (no consistency) to 1 (perfect). It is high on
buildings, roads and bare rock, and low on crops, water and woodland. Cells
below the coherence threshold (0.4 on the official route) are not measured.
A second measure, MintPy's *unwrapping consistency*, is delivered as
`quality_unwrapping_consistency.tif`; it is informational unless
`min_unwrap_consistency` is set above 0.

**Comparison ground, ordinary ground (farmland).** The yardstick every
figure is read against. It is ordinary ground around the site, away from
mapped mines (and a 2 km buffer around them), report areas (plus 500 m) and
any features found. Every figure is measured against it. The word used in
the report ("farmland") is set by `control_ground_label`.

**Detection limit.** The smallest steady sinking across an area that the
record would detect 4 times in 5. See section 6.3.

**DISP-S1 screening.** A quick way of processing a project (config
`backend: disp`) from OPERA DISP-S1, a ready-made NASA movement product for
North America. It takes a few hours and shows where movement might be. It is
for deciding where to look only: its report is never released to a client.
See "Processing routes" in section 1.

**Distributed-scatterer bias (DS bias).** On farmland no single strong
reflector dominates a cell. Over such ground, short image pairs carry a small
phase bias that looks like a downward rate on low-coherence ground. The
pipeline compares each area with farmland of the same coherence, which
cancels the bias.

**DRAFT, FINAL.** Every client report starts as a DRAFT. It becomes FINAL
only when nothing blocks release and a reviewer has signed off its exact
numbers. See section 7.3.

**Feature.** A patch inside the study area that the automatic search found
moving differently from ordinary ground. Features are graded by how unlikely
they are to be chance. See section 6.4.

**GNSS, GNSS anchor (GNSS tie).** GNSS is the general name for satellite
positioning, GPS among it. A permanent GNSS station is a fixed receiver that
records its own position every day, so it shows how the ground under it
moves. The GNSS tie adds the movement of the region as a whole to the
report's figures. Permanent GNSS stations within 40 km (Nevada
Geodetic Laboratory data) give the regional movement over the record's own
dates. Its uncertainty is the largest of the stations' own error, the scatter
between stations, and a regional floor measured from station rates within
100 km. At least 3 stations are needed; with fewer, the record is not
anchored and the study area reads "Relative only".

**Hard blocker.** Something that stops a report from becoming FINAL, such as
a missing sign-off or no firm recorded. Each one has a remedy; the list is in
section 7.3. A sign-off does not clear the others.

**InSAR.** Short for interferometric synthetic aperture radar: measuring
movement of the ground by comparing radar images of the same place taken at
different times. Changes of a few millimetres in the distance from the
satellite to the ground can be measured this way.

**Line of sight (LOS).** The slanted direction from the ground to the
satellite. The radar measures movement along this line. The report converts
it to vertical movement, assuming the ground moves straight up or down.

**Look, not a tested figure.** A figure worked out on the fly (in the
results map, for a short date window or a drawn polygon) without the
report's statistical test. Useful for exploring; not for quoting to a
client.

**Mini-stack.** Phase linking processes the images in groups of 10
("mini-stacks"); the patched MiaplPy used here joins neighbouring groups
through the images they share.

**Null (farmland null).** The spread of figures obtained by placing an area of
the same size and coherence on ordinary ground many times. A real movement
must stand outside it.

**Phase-linked route.** The official way of processing a project (config
`backend: miaplpy`), and the only one whose report can be released to a
client. It takes 1 to 2 days and about 300 GB of disk. Choose it in the map
builder as **Phase-linked**. See "Processing routes" in section 1.

**Phase linking.** Estimating each cell's movement history from many images
and similar neighbouring ground at once (MiaplPy). It recovers usable signal
on farmland, where simple image pairs fail. It is the longest step of a run.

**Posting and resolution.** Posting is how far apart the measurement points
are: 15 m on the official route. Resolution is how much ground each value
combines: about 90 m east-west by 240 m north-south on the official route
(about 41 x 42 m on the classic route; about 200 m on DISP-S1). Resolution is
always coarser than posting. Never describe 15 m as the resolution.

**Preflight.** The checks run before a run starts: disk, memory, logins,
weather-data access, the MiaplPy build and area names. A red line prevents
the run from starting.

**Project, project folder.** A project is one site: its outlines, its
settings and its results. The project folder holds everything for it: the
downloads, the processing files, the report and the records. It is in the
projects folder (`~/insar_projects` unless set otherwise), and its name is
the project's name in lower case with `_` for spaces. The project's settings
are in a separate configuration file, `configs/<name>.yaml` in the projects
folder. See section 9.1.

**Quality assessment (QA).** Six checks of every finished run against
published practice, written to `<project>/quality_assessment.json`. See
section 7.1.

**Reference point.** The point all raw measurements are first made against.
The report's figures are made against the comparison ground instead, so the
reference point's own drift cancels.

**Restart kind.** Which part of the processing a run starts from (Resume,
Whole pipeline, Products only and others). See section 8.2.

**Send rule.** The pipeline's recommendation on whether a report can be sent:
"send", "send with notes" or "hold for specialist". See section 7.2.

**Sentinel-1.** The European Copernicus radar satellites whose images the
program uses. They pass over the same ground about every 12 days, and radar
sees through cloud and in the dark. The records the program builds start in
2016. The images are free, from NASA's archive.

**Signal-to-noise (SNR).** A point's or area's movement divided by its own
uncertainty. A value of 1.96 or more "stands out"; about 5% of ordinary
ground does so by chance.

**Sign-off.** A reviewer's recorded confirmation that they have read the
report and checked its features. It is bound to the report's numbers: any
change to a number voids it.

**Study area, processing box.** The study area is the project area as a
whole. The processing box is the larger rectangle processed around it; its
margin supplies the comparison ground.

**Terminal.** A window where you type commands instead of clicking. For
this program it is the Ubuntu window (see "Before you start" in Start here).

**Ubuntu, WSL.** Ubuntu is a version of Linux, the system the program runs
on. WSL (Windows Subsystem for Linux) is the part of Windows that runs Ubuntu
inside your Windows computer. Open it as the **Ubuntu** app from the Start
menu. Everything inside it is stored in one large file on the Windows drive.

**Vertical.** Straight up or down. The radar measures along its slanted line
of sight; the report converts that to vertical, assuming the ground moves
straight up or down. Negative means sinking, positive means rising.

**95% range.** The range in which the true rate lies with 95% confidence,
from this record's own measured noise. See section 6.3.

---

## 3. The process at a glance

This chapter lists what the program does, stage by stage, during the day or
two of a run. Read it to understand what the progress screens are showing.

A project goes through these stages. The dashboard's live view shows each one
with its progress, elapsed time, estimated remaining time and log.

| Stage (as the dashboard labels it) | What happens |
|---|---|
| Checks before the run | the configuration, disk budget and logins are checked |
| Download Sentinel-1 data | one image per date is assembled from the parts that cover the box; each date gets 3 attempts |
| Orbit files | precise satellite orbits are fetched |
| ERA5 weather download | (ERA5 projects) weather data for the atmospheric correction, fetched in the background |
| Stack 01 to 12 | the images are aligned to each other (coregistration) and merged |
| Phase linking | MiaplPy estimates each cell's history; the longest stage, often most of a day |
| Stack 16: unwrap | the linked phases are exported as image pairs and unwrapped (converted to continuous movement) |
| Weather (HRRR) | (HRRR projects) the weather delay of each date from NOAA's HRRR model, about 2 minutes a date, before the troposphere step |
| MintPy steps | the time series is built: network inversion, atmospheric correction, removal of large-scale ramps away from the mines, elevation-error correction, and conversion to map coordinates at 15 m |
| Deliverables | rasters, point products and a summary are written |
| Quality assessment | the six checks and the send rule |
| Figures, panel and site histories | technical figures and the mining history of each dated panel |
| GNSS tie + client report | the regional rate is added; the client report, area summaries PDF and Google Earth file (KMZ) are built |
| Provenance + release status | a record of what produced the result; the log ends with FINAL or DRAFT |

A whole phase-linked run takes 1 to 2 days. Runs are detached: closing the
dashboard, the window or the terminal does not stop them.

---

## 4. Running a project, step by step

This chapter is the full detail behind "Your first project in 10 steps" in
Start here, for whoever runs a project and prepares its report.

### 4.1 Check the machine (once, and after any change)

Open an Ubuntu window (from the Windows Start menu, type `Ubuntu` and click
the **Ubuntu** app). Type this command and press Enter:

    insar doctor

It checks the logins (NASA Earthdata, Copernicus CDS), the ERA5 licence, the
software environments, the unwrapper, the patched MiaplPy, the real disk
budget and the memory. It must end with **Ready.** Each red line says how to
fix it. `insar doctor --quick` skips the test suite.

Record who prepares the reports; the report footer prints it, and a report
cannot be released without it. In the app, open **This computer**, fill in
**Firm**, **Prepared by** and **Contact** under **Company** and press
**Save**. Or, in the Ubuntu window:

    insar doctor --firm "Firm Name" --preparer "Your Name" --contact "you@firm.com"

Both write the same settings.

### 4.2 Gather the mine outlines

The pipeline must know where mines and other known ground movement are. It
keeps them out of the ramp fit (so real movement is not removed as
"atmosphere") and out of the comparison ground. Missing or out-of-date mine
outlines are the most common cause of misleading comparisons. When in doubt,
include the layer.

Sources, best first:

- The mine operator or the client: current mine plans.
- State records. In Indiana, the Indiana DNR Coal Mine Information System
  has annual mined-area polygons; the Indiana Geological & Water Survey layers
  stop at December 2016 (see `context/SOURCE.txt`). Elsewhere, the state
  geological survey or mining regulator.
- Federal records: the Office of Surface Mining (OSMRE). MSHA records give
  mine locations as points, not outlines.

Note where each layer came from and how current it is. A layer mapped before
the end of the record cannot show recent workings, and the report says so.

Accepted files: a zipped shapefile, KML or KMZ, GeoJSON, or a GeoPackage. The
content must match the extension. Limits: 300 MB per file, 1 GB unzipped,
1,000 files in a zip, 200,000 outlines and 10 million vertices. Shapefiles may
be in any projection; GeoJSON and KML must be in latitude/longitude or name
their projection. Statewide layers are fine: only what lies inside the
processing box is kept.

### 4.3 Build the project in the map builder

In the app, press **New project**, then **Open the map builder**. In the
terminal dashboard, press `b` on the projects screen. Or type this in the
Ubuntu window:

    insar builder

A map opens in your browser.

1. **Project area.** Draw it or upload its outline. The dashed rectangle is
   the processing box; its margin supplies the comparison ground.
2. **Mines and known movement.** Upload each layer under the right heading:

   | Upload as | For | Notes |
   |---|---|---|
   | Underground workings | longwall panels, room-and-pillar areas | |
   | Surface mines | pits, spoil, reclaimed ground | features on this ground are reported as "change on surface-mine ground" |
   | Other known movement | landfills, quarries, anything else | kept out of the comparison ground |
   | Dated mined areas | one polygon per panel or per year, with the year in the attribute table | choose the year field (and optionally a name field) to get the panel-by-panel movement history after mining |

   Keep underground and surface mines in separate layers. Fill in each
   layer's **Source, date** box (for example "Indiana DNR mine maps, 2023");
   the report credits it. Tick **report its numbers** on a layer to give it
   its own row in the report.
3. **Extra report areas** (optional). Each gets its own row in the report.
   Keep each small enough to fit about 10 times inside the comparison ground
   around it (in practice under about 20 km²), or widen the margin.
4. **Reference point** (optional). Leave it empty unless you know a stable
   spot; the pipeline then picks one more than 2 km from every mapped mine.
5. **Dates and route.** The end date defaults to a week ago, because weather
   data are published about 5 days late. Choose **Phase-linked** for official
   results.
6. **Check the satellite archive** (the **Check archive and GNSS** button).
   This lists the satellite tracks and how much of the box each covers, and
   the GNSS stations the tie will use. It warns below 3 stations.
7. **Create the project.** Type its name and press **Create project**. The
   configuration is written to `$INSAR_PROJECTS/configs/<name>.yaml` and the
   project appears in the app and the dashboard within seconds.

The builder refuses names that look like file names or placeholders
(underscores, colons, "Area 2"), sub-swath choices that cover less than 99%
of an area, and overwriting an existing project without confirmation.

The place search sends your text to OpenStreetMap, with the firm and contact
from the settings (see `PRIVACY.md`). Search for a town or road, never a
client's or site's name.

### 4.4 Run the preflight and start

In the app, open the project and press **Start**. In the terminal dashboard,
open the project (Enter) and press `s`. The first choice is the
**Recommended** restart; for a new project this is Whole pipeline. Choosing
it opens the preflight checks. In the app, the label reads **ready** when no
line fails; press **Start the run**. A failed line is marked XX; fix it and
press **Check again**.

| Line | Red means |
|---|---|
| Disk budget | the projected peak does not fit the real free space (section 9) |
| Products memory | the products stage needs more than 90% of the machine's memory |
| Earthdata login | the NASA archive refused the login |
| ERA5 | (ERA5 projects) the Copernicus CDS token is wrong or the licence is not accepted |
| ERA5 publication lag | (yellow, ERA5 projects) the end date is within about 5 days of today |
| Weather delays (HRRR) | the `raider-env` environment is missing, or a date the series keeps is before 15 Jul 2016 or the area is outside the continental US |
| Patched MiaplPy | the installed MiaplPy fails its check for this project's settings |
| Report-area names | (yellow) a name reads like a file name; release will be blocked until it is fixed |
| Other runs | (yellow) other projects are running and share this machine's disk and memory |

**Weather model.** New projects in the continental US use NOAA's HRRR weather
model (3 km, from AWS, no account or queue) for the atmospheric correction;
HRRR starts on 15 July 2016, so a record asked for from earlier starts that
day, and the config's header says so. Projects elsewhere, or on a machine
without the `raider-env` environment, use ERA5 from the Copernicus CDS (31 km,
from 1950, but its queue can hold a run for hours and it is published about 5
days late). The validation in `VALIDATION.md` was done with ERA5.

In the terminal dashboard's preflight screen, press `e` to edit the
configuration and `r` to check again (the app cannot edit the
configuration). Start is disabled while any line is red. To see the disk and memory a run will need without
starting it, choose **Size the run only**.

### 4.5 Follow the run

The live view shows each stage, its progress and the log; so does the
project's page in the app. You can close the dashboard or the app; the run
continues, and reopening either finds it again. Only `x` (Stop), or **Stop**
in the app, stops a run, together with every program it started. When a run ends
you receive a Windows notification and a line in
`~/.insar_pipeline_notifications.log`.

If the run fails, see section 8.

### 4.6 Review the results

1. **Read the verdict.** The projects screen's Verdict column shows the send
   rule and whether the report is DRAFT or FINAL (section 7.2). In the app,
   the project's card and its **Results** tab show the same.
2. **Open the client report** (`o`, or **Report** in the app) and read it as
   the client will. Section 6 explains every figure.
3. **Check every listed feature** against aerial imagery and dates. In a
   working mine area, a sharp patch can be spoil, fill or a new pit rather
   than subsidence.
4. **Check every report area** has a real reading. A "Not testable" area does
   not block release, but read its reason and be prepared to accept it. Do
   not shrink the client's area to make it testable.
5. **Explore in the results map** (`m`, or **Results map** in the app) if you
   need to look at particular places or date windows (section 6.7).
6. **List what still blocks release.** The app's **Results** tab lists the
   first few. For the full list, type in the Ubuntu window (with the path of
   the project's configuration file in place of `<config.yaml>`):

       python -m insar_pipeline.release <config.yaml>

   `[must]` lines block release; `[read]` lines go on the reviewer's list.

### 4.7 Sign off

Sign off only after reading the whole report and checking every listed
feature. If the send rule is "hold for specialist", the designated InSAR
specialist must review the reasons and be named at sign-off (section 7.2).

In the dashboard: open the project and press `r` to sign off. Enter your
name, tick each feature you checked, add notes and confirm. The dashboard
records the sign-off, rebuilds the report and, if nothing else blocks it,
builds the client package. If the rebuild changed a number, the sign-off no
longer matches and you are asked to sign again.

In the app: open the project and press **Sign off** (beside Report; shown
until the report is FINAL). The dialog asks the same: your name (the
preparer's, to start with), a tick for each feature, the specialist's name
for a held report, and notes. It lists first whatever still blocks a final
release however you sign. Press **Sign off, rebuild the report and
package**. The sign-off is recorded at once; the report
rebuild and the package follow in the background, and the Results tab says
how it went. The app waits for that rebuild before it quits.

From the Ubuntu window:

    python -m insar_pipeline.release <config.yaml> --sign "Your Name" --features 1,2,3 --notes "checked against 2024 imagery"

For a report held for a specialist, add `--specialist "Specialist's Name"`.
Then rebuild the report (Products only) and build the package.

The sign-off is stored in `deliverables/review.json` (earlier ones in
`deliverables/reviews/`). Editing the configuration is not a review.

### 4.8 Package and send

The sign-off, in the app or the dashboard, builds the client package once
nothing else blocks release. To build it yourself (for example when the
sign-off's package step did not run), press `k` on the project's live view in
the terminal dashboard (the app has no separate package button), or type in
the Ubuntu window:

    python -m insar_pipeline.package <config.yaml>

It refuses unless the report is FINAL, and writes
`deliverables/releases/<YYYYmmdd_HHMM>/` in the project folder (in the app:
**Files**, pick the project, **Open in Explorer**):

| Folder | Contents | Send? |
|---|---|---|
| `client/` | the report, figures, area summaries PDF, KMZ and workbooks | **Yes. This is what the client receives.** |
| `data/` | GeoTIFFs in the report's terms: vertical rate, its uncertainty, each cell's change at its average rate, signal-to-noise, line-of-sight velocity and cumulative displacement | Only if the client wants GIS files |
| `technical/` | internal records | Never |
| `README.txt`, `MANIFEST.json` | a description of every file, and a checksum of each | With `client/` |

Never send files straight from `deliverables/`: it also holds raw
line-of-sight products that read differently from the report. Release
folders are read-only. A package built with `--draft` goes to
`<stamp>_DRAFT/` and is never for a client.

### 4.9 Free the disk

Once a project's FINAL package is built, its processing files are deleted
automatically and its results kept (see section 9). To do it by hand, or for
a project where automatic cleanup is off, use the app's **Files** page
(**Keep results only**), the Storage screen (`g`) or `insar cleanup`
(section 9).

### 4.10 Reissuing a report

If the client already has a report and you rebuild it with different numbers,
the new report must say so. Before rebuilding, add to the project's
configuration:

    report_revision:
      number: 1                 # 1 for the first reissue, 2 for the next
      supersedes: "2026-09-28"  # the date of the issue it replaces
      changes:
        - "One plain sentence per change: what moved, from what to what, and why."

The report, PDF, KMZ and package README then print "Revision 1, replacing
the report issued 28 Sep 2026" and the changes. The note is part of the
signed numbers: editing it voids the sign-off. Take every number in it from
the rebuilt report.

---

## 5. The dashboard

This chapter describes every screen, button and key. Use it when you need to
find where something is done.

There are two ways in, and they show the same things from the same files: a
run started in one can be watched or stopped in the other. Most people use
the app window (5.1); the terminal dashboard (5.2) is needed only for the few
things the app does not do.

### 5.1 The app window

Double-click **InSAR Pipeline** on the desktop (Windows; `install/windows/setup.ps1`
puts it there), or type this in an Ubuntu window:

    insar app

It opens in its own window:

- **Projects:** every project on a map and as cards, with its state, the send
  rule, DRAFT or FINAL, and each report area as the client report reads it.
  Filter by running, needing attention or finished.
- **A project:** the run as six steps (get the data, line up the images,
  phase linking, unwrap, time series, products and report), a progress card
  with the time left for each stage that is working, and four tabs: every
  stage, the results, the log (the running step's own log during phase
  linking), and what happened while you watched.
- **The live map** fills in as the run makes data:

  | Layer | Appears |
  |---|---|
  | study box, area of interest, mines, report areas | from the start |
  | the satellite frame (the bursts found) | once the data search has run |
  | measurement quality, line-of-sight velocity | once the time series is geocoded |
  | measurement points, GNSS stations | once the products and the GNSS tie are built |

  The pictures are the products drawn on a map: a look, not the report's
  figures. The results map (**Results map** button, section 6.7) has every
  point and the figures.
- **Start** offers the recommended restart first, runs the preflight checks
  for a run that processes, and starts only when none fails, as `s` does
  below. **Stop** asks first.
- **Report** opens the client report; while it is a DRAFT it asks first.
  **Sign off** (beside Report, until the report is FINAL) records the
  sign-off, rebuilds the report and builds the client package (section 4.7).
- **New project** opens the map builder (section 4.3). **This computer** shows
  the disk budget, memory and the runs going, and **Company**: the firm,
  preparer and contact printed on reports and map exports, and the company
  logo for the exports (upload a PNG or JPEG).
- **Files** shows each project's folders and their sizes. Tick files or
  folders and press **Delete chosen**. What the pipeline needs is marked KEEP
  and cannot be ticked: the logs, releases, sign-off, run records, `context/`,
  `dem/`, `mintpy/geo` and the delivered point products. Nothing is deleted
  while a run is going. Anything in `deliverables/`, or over 50 GB, needs the
  project's name typed. Deleting is permanent (there is no recycle bin in
  Ubuntu); each deletion is listed in the project's `logs/deletions.jsonl`.
  **Keep results only** and **Delete project** (a slim archive is kept first)
  are the same as `insar storage` (section 9).
- **Map layers:** besides the outlines, the client rasters (vertical
  movement, vertical change, line of sight, signal-to-noise, uncertainty,
  the quality layers), the elevation-model error, land cover and water.
  Hover over a layer for what it is.
- **Export map** (on a project) prints what the live map shows, this view and
  these layers, on a page: Letter, Legal, Tabloid, ANSI C or D, A4 or A3,
  either way round, in colour or black and white, as a PDF or a 300-dpi PNG.
  The page carries the coordinates, a scale bar, a north arrow, the legend,
  your notes, the data credits and a title block with the company logo. A
  project whose report is not FINAL is printed with DRAFT across the map. The
  file goes to `<project>/exports/maps/` and opens.

**What a map export holds:**

| Item | What it is |
|---|---|
| Page | Letter, Legal, Tabloid (11x17), ANSI C (17x22), ANSI D (22x34), A4 or A3; landscape (panel beside the map) or portrait (panel below it) |
| Colour | colour, or black and white (grey ramps for the rasters, hatching and dashes for the outlines, the logo in grey) |
| File | a PDF, or a PNG at 300 dpi |
| Base map | satellite imagery, OpenStreetMap, USGS topographic, OpenTopoMap, light grey, or none |
| Map frame | latitude and longitude ticks, a scale bar (feet and miles, and metres), a north arrow |
| Panel | the title (and subtitle), the legend of every layer drawn, the notes (yours, then the automatic ones), the data sources |
| Title block | the company logo, the firm ("Prepared by"), the preparer ("By"), the project folder's name, today's date, the print scale and page, and the status: FINAL, or "DRAFT: not signed off" |
| DRAFT | across the map, with "not for client use: the report is not signed off", whenever the project's report is not FINAL, whatever the page says |
| Footer | the software version, "map export", colour or black and white, the page and orientation |

The file is named `<project>_<page>_<orientation>[_bw]_<YYYYmmdd_HHMMSS>.pdf`
(or `.png`). Beside it, a file of the same name plus `.json` records the
options chosen, your notes, whether the report was FINAL, the software version
and when it was written.

Closing the window never stops a run. **Quit the app** (This computer) stops
the app itself; runs carry on. Editing the config (`e`), and building the
client package on its own (`k`), are still done in the terminal dashboard.

### 5.2 The terminal dashboard

Type this in an Ubuntu window and press Enter:

    insar

You work in it with single keys, listed below and along its bottom line.

**Projects screen**

| Key | Does |
|---|---|
| Enter | open the project's live view |
| `b` | the map builder: draw or upload the area and mines, create the project |
| `m` | the results map in the browser (section 6.7) |
| `n` | a new area inside the terminal |
| `v` | results for any area and any dates |
| `f` | browse files |
| `g` | storage: every project's size, and freeing disk (section 9) |
| `r` | refresh |
| `?` | open this guide |
| `q` | quit (runs keep going) |

The **Verdict** column shows the send rule and DRAFT or FINAL.

**A project's live view**

| Key | Does |
|---|---|
| `s` | start a run; the **Recommended** restart is listed first |
| `x` | stop the run and every program it started |
| `p` | preflight checks |
| `e` | edit the project's configuration |
| `v` | results |
| `o` | open the client report (asks first when it is a DRAFT or held for a specialist) |
| `r` | sign off the report (section 4.7) |
| `k` | build the client package for a report that is already FINAL |
| `f` | files |
| Esc | back |

**Storage screen** (`g`): `k` keep results only, `d` delete a project, `r`
refresh.

---

## 6. Interpreting the results

This chapter explains every figure and word in the report, the results map
and the Google Earth file. The reviewer needs all of it; anyone who explains
a report to a client should read 6.1 to 6.6 and 6.9.

### 6.1 Units and direction

Rates are in inches per year (in/yr) with millimetres beside them. They are
**vertical**: the radar measures along its slanted line of sight, and the
report converts that to up or down. Negative means sinking; positive means
rising.

### 6.2 What every figure is measured against

Each figure is the area's movement **relative to ordinary ground of the same
coherence** around the site, **plus the regional rate** from the GNSS
stations. The comparison ground excludes mapped mines (plus 2 km), every
report area (plus 500 m) and any feature found, so an area is never compared
with itself.

### 6.3 Area figures

**Rate.** The area's average vertical movement per year over the whole
record.

**95% range** (the "likely range"). Where the true rate lies with 95%
confidence. It is built from patches of ordinary ground of the same size and
coherence placed all around the site, so it reflects this record's real
noise. It need not be centred on the printed rate. The words "sinking" or
"rising" are used only when the range excludes zero; otherwise the signed
figure is shown without a direction word.

**Reading.** One label, the same in the report, KMZ, workbook and dashboard:

| Reading | Means |
|---|---|
| Movement detected | The area differs from ordinary ground by more than ordinary ground differs from itself 95% of the time. |
| Within normal ground variation | It does not. This is a finding, not a failure; the detection limit says how large a movement could have been missed. |
| Within normal ground variation; moves with the surrounding region | Not different from its surroundings, but the region as a whole (from GNSS) is moving. The area moves with the region, not on its own. |
| Not testable: (reason) | No valid comparison was possible, so no range and no verdict. Reasons: fewer than 5 measured points; the comparison ground fits around the area fewer than 5 independent times; under 50% of the outline measured; one side almost unmeasured. |
| Not measurable: (reason) | The satellite did not image it, or fewer than 5 points could be measured (water, woodland, ground that changed). |
| Relative only | No valid GNSS anchor. Areas are still compared with the comparison ground, but the movement of the study area as a whole is not measured. |

**Detection limit** ("Reliably detects sinking of X"). Sustained sinking of X
or more across the area would be detected 4 times in 5. It is measured by
planting movements of known size into ordinary ground. The sinking limit is
printed; the rising limit is added only when it differs by more than a tenth.

**Signal ÷ threshold.** The area's movement beyond ordinary ground divided by
the 95% threshold for an area of its size. 1 or more means "Movement
detected".

**Totals over the record.** Two figures may appear:

- *Measured change*: the change the record shows, with its 95% range. The
  curve may change pace once a year, but only where that change of pace is
  larger than ordinary ground of the same shape shows; otherwise it is the
  average rate × years. This is the total to quote.
- *At the average rate*: average rate × years, printed beside the measured
  change when the two differ materially (5 mm or 5%). The per-cell GeoTIFF
  `vertical_change_at_average_rate_in_<d0>_<d1>.tif` in `data/` is this
  figure, not the measured change.

**Recent rate.** The rate over the last two full years, with its own range,
so a movement that has stopped or started is visible. If the range includes
zero, the report says the recent rate "cannot be told apart from no
movement".

**The study area as a whole** (for example "Full Satellite Imaging") is not
tested against the comparison ground, since it contains it. Its figure is the
satellite's relative figure plus the regional GNSS rate, with the GNSS rate
on its own line.

**Not compared with ordinary ground.** An area that could not be compared
still shows its rate, with the part of the range that can be stated, labelled
"not compared with ordinary ground", and no direction word.

### 6.4 Features

Features are patches the automatic search found inside the study area.

| Wording | Means | What to do |
|---|---|---|
| stands out clearly | beats chance in the search and is confirmed by the quality assessment's independent test | check imagery and dates before describing it |
| worth a look | beats chance in the search but is not confirmed | check imagery and dates; do not describe it as movement without that check |
| within chance | weaker patches that ordinary ground produces by chance | none; listed so nobody wonders about them |
| change on surface-mine ground | at least 30% lies on a surface-mine layer | treat as mining activity unless shown otherwise; where the ground was dug or filled by metres, no rate is given |

How the search decides: the same search is run on 199 simulated copies of the
site's own comparison ground. A patch beats chance only when at most 4 of the
199 copies produce one as strong (a chance of 1 in 40 or less). When none
does, the report says "a chance of less than 1 in 200". With too little
comparison ground (under 50 km², or under a quarter of the area searched), no
feature is called beyond chance.

Further rules in the feature text:

- **Location.** Mining causes are mentioned only when a feature lies within
  500 m of a mapped mine edge. Farther away, the report gives the distance and
  says the mapped mines do not explain it.
- **Timing.** "It began around <year>" appears only when the years before
  are within what ordinary ground does; a feature moving from the start "was
  already sinking when the record starts".
- **Outline.** A feature's outline is drawn from the same measurements as its
  figures, so its figures read somewhat stronger than the true movement. The
  report says so.

### 6.5 Maps and charts

**The map** colours movement relative to the comparison ground. Grey is the
band ordinary ground reaches (its 2.5% to 97.5% range). About 5% of ordinary
ground is coloured by chance; the caption gives that expected area in km².
Hatching inside a report area marks ground that was not measured.

**The charts.** The thick line starts at zero and shows the average rate (or,
for a feature, the fitted curve where its change of pace passed the test).
The darker band is how far ordinary ground's average rate reaches: the line
ends outside it exactly when the table says "Movement detected". The lighter
band adds the noise of a single date; about 1 date in 20 outside it is
normal.

### 6.6 Limitations to keep in mind

- **East-west movement.** The radar looks sideways, so east-west movement
  also appears in the measurement. The report prints the conversion factor for
  the track used and, at mine sites, the resulting flank error (about ±28%).
- **One cell is not an area.** A single point's rate is much noisier than an
  area's. Quote point uncertainty from
  `vertical_movement_uncertainty_1sigma_<unit>_per_yr.tif`, not the formal
  uncertainty. Neighbouring points share ground and are not independent.
- **Trees.** The C-band radar sees the canopy, not the ground beneath it.
  Wooded cells (from the ESA WorldCover map) are left out of the comparison
  ground, each area's tree share is stated, and a feature a quarter or more
  wooded carries a caveat.
  - **Left out of the products by default** (`mask_wooded`, since 21 Oct
    2026): maps, rasters, points, KMZ, viewer and area figures. Woods carry
    less of the farmland bias that the relative figures remove, so their
    values read as rising. The quality assessment still uses every cell.
  - **Movement under trees** can therefore go unseen in the products. Where it
    matters, set `mask_wooded: false` for that project and read the wooded
    cells with the caveat.
- **Water and crops** lose coherence and are often unmeasured.
- **Mine outline currency.** Workings newer than the mine layer cannot be
  recognised as mining. The report states each layer's date.

### 6.7 The results map

Press `m` on the projects screen (or run `insar viewer`). A map opens in the
browser for any project whose client KMZ has been built. It runs on this
computer only.

**Choosing dates.** The bar at the top picks the window: the whole record, the
last 12 months, the last 2 years, any calendar year, or any two images
(choose them from the lists, or drag across the strip of images). Everything
on the page follows the window. On the strip, red ticks are images the record
left out (hover for the reason) and shaded spans are gaps of more than 36
days. A window needs at least 6 images over at least 3 months.

**Kind of figure.** A label beside the dates says which kind you are reading:

- **Report figures (whole record)**: the client report's own numbers.
- **Look, not a tested figure**: any shorter window, worked out on the page
  from each point's series. For exploring only.
- **Tested**: a window run through the report's test (`insar results`).

The regional GNSS rate is the whole record's for every window.

**What you can do:**

- **Colour the points** by vertical movement, line-of-sight velocity, change
  over the window, signal-to-noise, measurement quality or unwrapping
  consistency. "Hold the colour scale" keeps the scale fixed while you change
  dates.
- **Click a point** for its figures and a chart (hover for date and value).
  The chart shows the point's cell, the 3 x 3 cells around it, the average-rate
  line, and the band where ordinary ground of the same noise would lie.
- **Draw a polygon or rectangle** for the average of the points inside, its
  chart, a histogram against ordinary ground, the rate in each calendar year
  and the share standing out. Name it at the top of its panel, then export
  it (table below), or **Save as report area** (the configuration is backed
  up first).
- **Pin to compare** puts up to four points or polygons on one chart.
- **Download** any chart as PNG, SVG or CSV; **Table** shows the numbers.
  Downloads state the dates, the kind of figure and the report version, and
  are marked DRAFT until the report is signed off.
- **Share a view** by copying the address bar; it holds the project, dates,
  layer, map position, polygons and selection.

A polygon's figures are a plain average, not a tested figure. To get report
figures for a drawn area (rate, 95% range, detection limit), save it as a
report area and rebuild the client report (Products only).

**Polygon exports.** Each follows the dates chosen at the top. The file goes
to the browser's downloads folder (on Windows, your Downloads folder), not
into the project. A window other than the whole record adds
`_<first>_<last>` to the file name.

| Button | What you get |
|---|---|
| **Points CSV** | one row per ~100 m point inside the polygon, then a last row "polygon average" |
| **CSV, every 15 m cell** | the same, one row per measured 15 m cell. It replaces Points CSV once the page has measured the cells inside (up to 40,000 cells, about 9 km² of fully measured ground) |
| **Points GeoJSON** | the ~100 m points with their figures, for GIS (it stays at the ~100 m points when the cells are shown) |
| **KMZ** | the polygon (its average and chart in its balloon) and the ~100 m points inside, with the client KMZ's point balloons |
| **KMZ, every 15 m cell** | the same with every measured 15 m cell inside instead of the points (up to 40,000 cells, about 9 km²); the first one can take a minute |

The CSV files are in **inches**. They start with `#` lines: the project and
polygon name, the number of points, the dates and the kind of figure, the
report's numbers hash, the software version, and what each column means.
The columns are `latitude`, `longitude`, `vertical_in_per_yr`,
`line_of_sight_in_per_yr`, `line_of_sight_change_in`, `signal_to_noise`,
`signal_to_threshold`, `temporal_coherence`, `unwrapping_consistency`,
`too_fast_to_measure`, then one `vertical_in_<date>` column per image (the
point's own cell, zeroed on its straight-line fit at the first image). A point
too fast to measure has no figures. Both KMZ files name the dates, the kind of
figure, the numbers hash and the software inside.

### 6.8 The Google Earth file (KMZ)

Open it in Google Earth Pro (desktop). It opens on the study area. Every
figure in it is read from the client report, so the two cannot disagree.
Click the file's own name at the top for its description: the report's
status (and what blocks it while it is a DRAFT), any revision note, the
limitations and a guide to the layers.

What appears in Google Earth's list, top to bottom:

| Folder or layer | On when the file opens? | What it holds |
|---|---|---|
| Legend (bottom left) | on | the colour scale of the points and the 15 m cells |
| **Report areas** | on | each area's outline. Click it for its rate, measured total, 95% range, detection limit, signal ÷ threshold and reading, as the report gives them, and its chart |
| **Features** | on | the report's numbered features: outline, the report's words, figures and chart |
| **Measurement points: vertical movement (every ~100 m)** | **off**: tick it to see the points | one point about every 100 m over the whole study area. Colours follow the report map; lighter points are within normal variation for a single point. It was turned off because it made Google Earth slow to open |
| **Feature points** | **off** | a folder for each feature ("Feature 1 - points" and so on) with extra points inside it. Present only when the report lists features |
| **Map layers (one at a time)** | the first layer is **on** | only one shows at a time: **Vertical movement, every measured 15 m cell** (on when the file opens); line-of-sight velocity (relative); line-of-sight cumulative displacement (relative); signal-to-noise ratio; the report's map (smoothed; grey = normal variation). Each has its own legend |
| **Tree cover** | off | stripes mark wooded ground, where values are less certain. Present only when the project has tree cover |
| **GNSS stations** | on | each station, with its vertical rate over the record |

**A point's balloon** (click a point) shows, for that point's own 15 m cell:

- its position ("Point 40.12345 N, ...") and the area it lies in;
- the vertical rate per year, in inches with millimetres beside it
  ("relative to the surrounding farmland" when the record is not tied to
  GNSS);
- the total at the average rate over the record, with its 95% range;
- the 95% range of this point with the regional movement (when tied to GNSS);
- the rate compared with ordinary farmland, with its 95% range;
- signal ÷ threshold ("stands out" at 1 or more) and the signal-to-noise
  ratio (1.96 or more stands out);
- the line-of-sight rate and the line-of-sight change from the first image to
  the last;
- measurement quality (temporal coherence) and unwrapping consistency;
- a chart of its movement over the record, averaged over the 3 x 3 cells
  around it;
- a footnote: what ground each value combines (about 90 m x 240 m), that
  vertical is assumed, and that single points are noisy.

A point too fast to measure says so instead of giving figures.

**Units.** The balloons and charts are in inches, with millimetres beside
them, as the report is. The chart data files inside the KMZ
(`charts/area<n>.csv` and `charts/feature<n>.csv`, with the columns `date`
and `area_vertical_mm` or `relative_vertical_mm`) are in **millimetres**. To
open them, rename a copy of the KMZ to `.zip`. Divide by 25.4 to compare
them with the report's inches.

A KMZ named `..._DRAFT.kmz` belongs to a report that is not yet FINAL; one
named `..._SCREENING_...` comes from the DISP-S1 screening route. For every
15 m cell of a chosen place and dates, use the results map's
**KMZ, every 15 m cell** (section 6.7).

### 6.9 Common misreadings

| Misreading | Correct reading |
|---|---|
| "Within normal ground variation" means no movement. | It means no movement larger than the detection limit. Smaller movement may exist. |
| A coloured point on the map is movement. | About 5% of ordinary ground is coloured by chance. Only a cluster, or an area's tested reading, means anything. |
| 15 m is the resolution. | 15 m is the spacing of points. Each value combines about 90 m × 240 m of ground. |
| A short-window figure in the results map can be quoted. | It is a look, not a tested figure. Quote the report, or run the window through `insar results`. |
| A feature that "stands out clearly" is subsidence. | It is a statistical finding. Check imagery and dates; spoil, fill and new pits produce the same signal. |
| The study area figure is the same kind of figure as a report area's. | It is not tested against ordinary ground; it is the relative figure plus the regional GNSS rate. |

### 6.10 The local change layer (vendor-style, optional)

The official record fits one continuous series through every image pair. The
local change layer works the way the vendor's point product does: for every
point, it counts only the 12-day steps whose own quality passes, and a failed
step adds nothing. It is built from a separate full-resolution processing of
the same images (about 16 × 14 m). Each step is read against the point's
surroundings, so no unwrapping is needed.

**It is a screening layer only.** On 22 Oct 2026 it failed its validation
against the vendor at Site D, in both configurations tried, against criteria
fixed in advance (see the end of this section). Don't quote it as a finding.

What it is good for:
- pointing to places to check by other means (imagery, long pairs, crop
  history, mine records);
- comparing with vendor exports like for like.

What it cannot do:
- say whether a whole area is moving. Each step is read against a smooth
  reference about `local_change_sigma_km` wide (4 km), which removes anything
  broader: weather and orbit, but also real regional or mine-wide movement;
- count movement during the steps it rejects (snow, crop growth, wet spells);
- remove the farmland drift. The 12-day steps are where the drift is largest,
  and the step quality check doesn't remove it. Near two Site D towns, cropland
  read about 8-10 mm/yr lower than the few built-up points the check kept
  (−2.9 and −2.1 against +7.2 and +5.5 mm/yr). That is about half the pit-edge
  signal the layer raised. It was a check made after the fact, and few built-up
  points pass, so the size is uncertain. So compare a spot only with ground of
  the same land cover and crop history, or confirm it with long pairs (below).

How partners work: each point is averaged with its "partners", the neighbours
whose brightness behaves alike. They are chosen afresh for each ~2-year period
(`local_change_partner_window_days`), because crops rotate and the ground
changes. Bright, steady points (persistent scatterers) may count without
partners. But at these looks speckle alone spreads the brightness by about 0.3
(sd ÷ mean), so the 0.15 test passes almost nothing: 0.04% of points at
Site D. In practice that path is off.

Its rate is "while observed": counted movement divided by counted time. That is
the vendor's definition, and it is larger than a yearly rate when half the steps
are rejected.

To make it:

1. Copy the project's config with:
   - `stack_range_looks: 4` and `stack_azimuth_looks: 1`;
   - `stack_virtual_merge: "False"`, because the layer reads the real merged
     images;
   - a new `project_dir`, `purpose: experiment` and the dates you want.

   The area must span at least two bursts north-south; the run plan refuses
   one that doesn't.
2. Process to the merge, on either route:
   - **Phase-linked-style merge (smaller):** set `backend: miaplpy`,
     `stack_cleanup_coreg_after_filter: false` and
     `miaplpy_delete_merged_slc_after_load: false`, then run
     `python scripts/run_pipeline.py <config> --to-step merge_reference_secondary_slc`.
     It frees each date's aligned bursts as its merged image is written: about
     241 GB for 275 dates at Site D. The layer then forms the 12-day pairs
     itself from the merged images.
   - **Stack route:** set `backend: stack` and `stack_num_connections: 1`, then
     run `python scripts/run_pipeline.py <config> --to-step filter_coherence`.
     The merged images sit beside the aligned bursts until filtering, about
     475 GB at Site D. The plan now counts them.
3. Point `local_change_era5` at the official project's `mintpy/inputs/ERA5.h5`
   (same track and dates). Optionally point `local_change_dem_error` at the
   official `mintpy/geo/geo_demErr.h5`, which removes each step's DEM-error
   residual. A 25 m error, such as a new spoil heap, adds about 4 mm to a
   12-day step.
4. Run `python scripts/run_local_change.py <config>`. Options:
   - `--compare name=<export.csv> ...` sets it beside the vendor's figures;
   - `--box name=lat,lon,radius_m ...` and `--period YYYYMMDD-YYYYMMDD ...`
     sample stable ground or a feature over chosen years;
   - `--tag sigma8km --sigma-km 8` re-runs only the steps with an 8 km
     reference, into `local_change/variants/sigma8km/`, to test whether a
     result depends on the reference width.

   Comparisons are saved to `compare.json` beside the steps.

It writes `<project>/local_change/`:
- the rate (`local_change_rate_mm_yr.tif`);
- the share of steps kept;
- the counted displacement and counted steps;
- each pixel's noise (`local_change_noise_1sigma_mm_yr.tif`);
- the t statistic (`local_change_t.tif`);
- the rate masked to the significant pixels
  (`local_change_rate_significant_mm_yr.tif`): a two-sided t-test at the
  probability of `local_change_min_snr` Gaussian sigma, with at least
  `local_change_min_steps` counted steps. A map cell shows only where most of
  its radar pixels are significant. Inside the map's area the map and the
  pixels agree (Site D: 1.34% of map cells against 1.44% of the pixels that
  can be significant, configuration 2; 1.92% against 2.04%, configuration 1).
  The `share_significant` in `local_change.json` (2.04% and 2.76%) counts the
  whole processed window, about three times the map's area. Judge the share
  from the JSON, and remember which area it covers.

**Read only the significant map, and only as places to check. Compare its share
with what chance gives** (`local_change.json` → `summary`: `share_significant`
over `eligible_pixels`, against `expected_by_chance`). Pixel by pixel, the unmasked
rate is mostly noise: at Site D over 20 months each pixel's noise was about
±1 in/yr while observed. That is the size of the field-by-field patchwork the
unmasked map shows.

The group differences in `--compare` are not datum-free. Each group keeps its
own steps, so the differences depend on what they are read against. Two
figures from Site D, measuring different things:
- moving the reference point shifted the official series' group differences
  by up to 0.27 in/yr, over the vendor's 20 months
  (`scripts/compare_vendor_points.py --reference-box`);
- changing this layer's reference width from 4 km to 8 km moved its group
  differences by up to 0.56 in/yr (configuration 2, below).

Quote that sensitivity with any difference.

**Validation result (22 Oct 2026).** The criteria were fixed in advance (plan
of 22 Oct 2026). Both configurations failed them:
- Configuration 1 (partners chosen once over the 10 years). The share of steps
  kept was far from the vendor's everywhere (2% against 77% on buildings), and
  two of four group differences missed the vendor's by more than 0.20 in/yr.
- Configuration 2 (partners per ~2-year period, persistent-scatterer path). The
  four group differences came within 0.20 in/yr of the vendor's, but that is
  in-sample: the settings were revised on the same dates. The kept share still
  missed (5% against 77% on buildings). A town control did not stay still
  (+6.4, +4.1 and −4.8 mm/yr while observed over three periods). And the group
  differences moved by up to 0.56 in/yr between a 4 km and an 8 km reference.

So the layer stays a screening tool, with no third round of tuning. Use it
beside the official record to point to places to check by other means.

**Checking a feature with long pairs** (`feature_check.py`,
`scripts/run_feature_check.py`)

The farmland drift is a short-pair effect: largest in 12-day pairs, fading in
pairs 6 months to a year long. So a feature can be checked against a ring of
ground around it with pairs about 6 months (182 days) and 1 year (365 days)
apart. That needs no partners, no step quality check and no weather model.
Each pair's change is the phase of the feature against its ring. The check then
fits one rate per calendar year, with a jackknife error.

Run `python scripts/run_feature_check.py <config> --area name=lat,lon,radius_m,ring_in_m,ring_out_m`,
or `--vendor name=<export.csv>` for a vendor's points. It needs the merged
full-resolution images, as the layer above does. It writes
`<project>/feature_check/feature_check.json`.

Limits:
- a pair holds ±14 mm without ambiguity: ±28 mm/yr on 6-month pairs and
  ±14 mm/yr on 1-year pairs. Faster movement aliases on the 1-year pairs, which
  is why both are computed;
- a year the kept pairs cover for less than half a year, or can't tell apart
  from another year, is "n/a", never 0;
- the error on 1-year pairs comes out too small (on pure noise the rate passed
  twice its error in 10-11% of years, not 5%). Read it as a lower bound. On
  6-month pairs it is about right (4-5%) only while nearly every pair is kept.
  Where many pairs drop, as on ground that loses coherence, it is too small as
  well: about 7% with half the pairs kept, 8-10% with a quarter and 11-13%
  with 5-10% kept (the pit edge at Site D kept 5%). Read it as a lower bound
  there too, and look at `pairs_kept`, `pairs_used` and `groups_per_year` in
  the record;
- a year shown with a value but "±n/a" has no error because a refit without
  one block can't solve it. It is weakly determined, so don't judge it.
  `cover_used_years` gives each year's cover by the pairs actually used, so it
  shows why a year with enough cover in `cover_years` still reads "n/a";
- the error rests on 40 blocks of start dates (`--groups`, recorded as
  `jackknife_groups`). The 22 Oct 2026 check used 10 blocks with the earlier
  code; its errors can be reproduced only with that code. Its verdicts didn't
  rest on the errors.

Result at Site D (22 Oct 2026, criteria fixed in advance): both town controls
passed. On 6-month pairs every year stayed within ±3 mm/yr (the test allowed
±4), and the 1-year pairs agreed within 3 mm/yr. The one 1-year value past
±4 was a town's 2026 partial year (−4.2 mm/yr, against −2.7 on 6-month pairs).
But farmland, the spoil patch and the pit edge lose coherence within about
6 months, so long pairs can't see them. Farmland kept 4 of 257 pairs, the spoil
patch none, and the pit edge 14, with no year from 2023 on covered. The check
works on coherent ground; it can't yet confirm or reject a farmland feature.

---

## 7. Quality checks, the send rule and release

This chapter is for the reviewer. It explains how the program decides
whether a report may go to a client, and what must be settled before it can.

### 7.1 The six quality checks

Every finished run is checked against published practice. The results are in
`<project>/quality_assessment.json`. Some checks are **expected to flag on
farmland**; the figures are built to allow for those flags.

| Check | What it asks | Flags when | Expected on farmland? |
|---|---|---|---|
| Reference stability | Does the reference point drift against other stable ground? | drift over 1 mm/yr | Yes. Figures relative to farmland are unaffected. |
| Rate precision | How well does one cell's rate repeat? | worse than 2 mm/yr (median) | Yes. Area figures average many cells and use the farmland spread as their uncertainty. |
| Single-epoch noise | How noisy is one date's reading far from the reference? | noisier than 10 mm | No. |
| Coverage | How much of the area was measured, and is the coherence threshold above the noise? | under 70% measured, or threshold inside the noise level | No. |
| Network | Do the image pairs form one connected network with enough redundancy? | broken into pieces, or fewer than 3 connections per date | No. A broken network also blocks release. |
| Distributed-scatterer bias | Does low-coherence farmland read more negative than high-coherence ground? | a difference over 3 mm/yr | Yes. Comparing with farmland of the same coherence removes it. |

On the DISP-S1 route there is a seventh check, **stitch**: were the OPERA
granules joined cleanly? It is not expected to flag on farmland.

The QA also records an overall verdict (MEETS PRACTICE, USABLE WITH
LIMITATIONS or REVIEW) that only counts flags. Use the send rule instead: it
considers **which** checks flagged.

### 7.2 The send rule

The send rule is shown in the dashboard's Verdict column and recorded in
`deliverables/client_report.json` under `send_rule`, with its reasons under
`send_rule.reasons`.

| Value | Means | What you do |
|---|---|---|
| **send** | nothing flagged | Review and sign off. |
| **send with notes** | only flags expected on farmland, plus recorded limits (a dropped date, reliable coverage under 70%, a requested unwrapping-error correction that MintPy skipped) | Review and sign off. The report explains each note under "Reading the numbers"; a skipped correction is on the release list and in `quality_assessment.json` → `unwrap_correction` only. |
| **hold for specialist** | anything else (below) | Do not sign off alone. Refer the project to the designated InSAR specialist (section 11). The report stays DRAFT until a sign-off names the specialist who reviewed the reasons. |

"Hold for specialist" results from any of:

- a flag not expected on farmland (single-epoch noise, coverage, network,
  DISP stitch), or a check that could not run;
- a coherence threshold inside the measured noise floor;
- too little comparison ground, or a report area that cannot be compared with
  it;
- an image date missing without a recorded reason;
- map positions more than 10 m off mapped water edges, or one half of the
  area more than one 15 m cell off when the two halves really differ;
- a figure that moves by more than its uncertainty when the ramp removal or
  the farmland matching is changed;
- more than 5% of a report area changing too fast to measure;
- no GNSS anchor.

A check that did not run reads "not run", never "passed".

The QA runs before the GNSS tie, so the report stage re-evaluates the rule
with the tie. The client report's rule is the final one. If the QA has been
re-run since the report was built, the dashboard marks the rule "(report out
of date)" and release is blocked until the report is rebuilt (Products only).

### 7.3 DRAFT and the release gate

Every client report starts as a **DRAFT**: the page carries a watermark and a
list of what blocks it, the KMZ is named `..._DRAFT.kmz`, and the area
summaries PDF is marked DRAFT. It becomes FINAL only when there are no hard
blockers **and** a reviewer has signed off these exact numbers.

**Hard blockers** (the report cannot be released):

| Blocker | Remedy |
|---|---|
| no firm or preparer recorded | `insar doctor --firm "..." --preparer "..."` |
| the image network is not one connected piece | refer to the specialist (section 11) |
| the report comes from the DISP-S1 screening route | run the phase-linked route |
| a report area is named like a file (for example `project_zone`) | give it the client's name in `report_areas` or the builder, then Products only |
| under 50% of a report area measured, or the report does not say | rebuild (Products only); if it persists, the area cannot be reported as drawn |
| a GNSS station inside the area disagrees with the radar | refer to the specialist |
| a part of the report a printed figure or the send rule depends on failed (`report_part_failed`), such as the single-point uncertainty or the send rule re-evaluated with the GNSS tie | Products only; if it fails again, refer to the specialist |
| the last time-series (MintPy) run did not finish | Resume, then rebuild |
| the report predates the quality assessment (`report_older_than_qa`) | Products only |
| stale inputs: the QA, summary, uncertainty model or GNSS tie is older than the velocity product, or for another record window | Products only (it re-runs the tie and the report) |
| a listed feature has not been ticked at sign-off | sign off again, ticking each feature checked |
| no sign-off, or a sign-off for an earlier build of the numbers | sign off (section 4.7) |
| a reissue that does not say so (`reissue_without_note`) | add a `report_revision` note (section 4.10), then rebuild |

Other report parts that failed are listed under "Left out of this build" on
the report.

**Reviewer items** do not block release but go on the reviewer's list, to be
read and accepted before signing off: each flagged QA check; a send rule other
than "send"; a disagreeing DISP-S1 cross-check (when `cross_check_disp` is
on); a report area that is "Not testable" (`area_not_testable`); and GNSS
stations not screened for equipment changes and earthquakes
(`gnss_steps_not_screened`, cleared by Products only when the screening data
can be read).

To list both kinds from a terminal:

    python -m insar_pipeline.release <config.yaml>

### 7.4 Tracing a reported figure back to its source

When someone asks where a reported figure came from, follow it back through
these records in the project folder. Cleanup keeps all of them. A deleted
project's archive keeps the JSON records, but not `mintpy/` and not
deliverable files over 50 MB (a large point file among them). Figures in the records
are in millimetres (divide by 25.4 for the report's inches).

| Step | File | What to read |
|---|---|---|
| 1. The figure | `deliverables/client_report.json` | Features are under `features`, in the report's order (Feature 1 first). Each entry holds its rate against the comparison ground (`vertical_relative_mm_yr`), its 95% range (`rate_range95_mm_yr`), its location (`lat`, `lon`, `area_km2`, `bbox`), its outline (`ring`, longitude and latitude) and the comparison ground's name (`control_ground_label`). The comparison ground itself is described under `comparison_ground`; report areas are under `areas`. |
| 2. The points | `deliverables/points_attributes.gpkg` | Open it in a GIS and select the points inside the feature's outline. Each point has its radar pixel (`radar_pixel_id`; cells sharing one share one measurement), its position, and its rates in mm/yr: `mean_velocity` (against the reference pixel) and `mean_velocity_relative` (against ordinary ground of its coherence). |
| 3. The time series | `mintpy/geo/geo_velocity.h5` and the reference pixel | The rate grid the points are taken from. The reference pixel is in `deliverables/provenance.json` (`reference_point_lalo` as asked, `reference_pixel` as used) and in every delivered GeoTIFF's metadata. |
| 4. The images | `deliverables/provenance.json` | `inputs.scenes_used` lists the Sentinel-1 granule IDs, `inputs.dates_processed` the dates, `inputs.dates_excluded_before_inversion` any dropped dates, and `inputs.pairs` the image pairs. `builds` names the software versions that made the record, and `config.snapshot_sha256` ties it to `deliverables/config_snapshot.yaml`. |
| 5. The regional movement | `deliverables/gnss_tie.json` | `stations` lists each station (`site`, `distance_km`), its data span (`data`, in years), its rate over the record (`vertical_over_record_mm_yr`, with `span_years`), its overlap with the record (`overlap_years`) and whether it was used (`used_for_background`). `background` is the regional rate added to the figures; `record_window` gives the dates. |

**The mine outlines.** Where each mine layer came from, and how current it
is, is recorded in the project's configuration under `mine_layer_sources`
(one line per kind of layer, such as "Underground mine workings: Indiana DNR
mine maps, 2023"). It is what was typed in the builder's **Source, date** box;
when that box was left empty, it is the uploaded file's name and date, or
"source and date not recorded". The report prints the same lines. The
uploaded originals are kept in `<project>/context/sources/`.

---

## 8. Troubleshooting

Use this chapter when a run stops, a button does not work or a message is
unclear. Start with 8.1; most problems are solved by the restart it
recommends.

### 8.1 When a run fails

1. **Open the project in the dashboard.** The header gives the plain cause
   and a **What now:** line (both from `<project>/run_result.json`). The stage
   with a red ✗ is where it stopped. A yellow line counts the warnings the run
   logged; a finished run can carry warnings worth reading too.
2. **Press `s` and take the Recommended restart**, unless the What now line or
   the table in 8.3 says otherwise. Every stage skips work it can verify on
   disk, so a restart never redoes finished work.
3. **If the same step fails twice in the same way**, stop and refer the
   project to the specialist (section 11). Repeating it further will not help.

The log of each run is `<project>/logs/<kind>_<time>.log`; the What now line
names it. A long log rolls over to `.1`, `.2` and `.3` (section 8.6).

### 8.2 Restart choices

From the most work to the least:

| Choice | Starts from | Use when |
|---|---|---|
| Resume | whatever the disk shows | almost always: it picks the right one of the others |
| Whole pipeline | the downloads | a new project; on a finished phase-linked project it re-plans from the start, and the dashboard warns you |
| Phase linking onward | the coregistered stack | coregistration finished; phase linking or later failed |
| Re-unwrap | the exported pairs | some pairs failed to unwrap, or an earlier Re-unwrap was interrupted |
| MintPy onward | the unwrapped pairs | the time series (MintPy) failed |
| Products only | the finished time series | a deliverable, QA, figure, GNSS tie or report step failed; after a configuration change that affects only the report; after sign-off, to rebuild the report |
| Size the run only | nothing (downloads nothing) | to see the disk and memory a run will need |

An unsuitable restart is refused with a plain message (for example "Phase
linking onward" before coregistration has finished).

### 8.3 By the stage that failed

| Stage | Usual cause | What to do |
|---|---|---|
| Checks before the run / config | a mistake in the configuration, or too little disk | Fix what the message names (`e` on the preflight edits the configuration). Free disk space (section 9). |
| Download Sentinel-1 data | login refused; network failure; a date the archive cannot supply | Login: run `insar doctor`. A NASA or ASF outage shows there as a warning, not a wrong password: wait and Resume. Network: Resume. A date that fails 3 times stops the run with the dates named: wait a day and Resume, or accept the loss with `allow_missing_dates: ['YYYYMMDD']` in the configuration. |
| Download (footprint) | the chosen sub-swaths cover under 99% of an area | Add the sub-swath the message names to `stack_swaths`. |
| ERA5 weather download | CDS token or licence | Fix with `insar doctor`. The run carries on; MintPy fetches the data later. |
| Stack 01 to 12 (coregistration) | out of memory or disk; or the alignment network between dates fell apart (Stack 07) | Memory or disk: free it (section 9) and Resume. Network: the message names the dates; the remedy changes processing settings, so refer it to the specialist. |
| Phase linking: load data | an interrupted load left a partial file | The message asks for `miaplpy/inputs/slcStack.h5` to be deleted. Make sure no run holds the project, delete that one file (Files screen, or `rm` in a terminal), then choose Phase linking onward. |
| Phase linking: phase linking | a worker stalled (usually killed for lack of memory), or settings changed since linking began | Stall: lower `miaplpy_num_processors` and Resume; finished patches are kept. Changed settings: the message names them; put them back. Re-linking from scratch takes most of a day; refer that decision to the specialist. |
| Phase linking: patch check | the installed MiaplPy is not the patched build | `bash patches/miaplpy/build_miaplpy.sh miaplpy`, then Resume. |
| Phase linking: datum gate | the check joining the mini-stacks tripped | Refer to the specialist; the technical background is in `INSAR_RUNBOOK.md` section 11. |
| Stack 16: unwrap | some pairs failed to unwrap | The pairs are set aside in `stack/quarantine_unwrap_timeout/`. Resume (it chooses Re-unwrap). If it fails again, refer to the specialist. |
| MintPy steps | weather data not yet published for the last dates; the ramp mask left too little ground; the network fell apart | Weather data: wait a week and Resume, or set an earlier `end_date`. Ramp mask: the message lists the remedies (such as `deramp_buffer_m` or a wider box). Network: refer to the specialist. |
| Deliverables, Quality assessment, Figures, GNSS tie + client report, Provenance | a product step failed | Products only. If it fails twice, refer to the specialist. |
| Validation against ground truth | validation experiments only (`purpose: experiment`): the GNSS download failed or the protocol file is missing | The products are unaffected. `insar validate <config> --protocol <file>` repeats the validation without a run. |
| Disk watch | free disk fell below 20 GB, so the run stopped itself | Free space (section 9), then Resume. |

### 8.4 Other situations

**State "interrupted".** The run ended without reporting, usually because
Windows restarted WSL (for example for Windows Update) or the computer ran out
of memory. The header says which and where. Resume.

**"Already running".** Another run, a cleanup or a sign-off rebuild holds the
project. Wait, or stop it from the dashboard. Never delete
`<project>/.run.lock` by hand; it is released when the program ends.

**Start is disabled on the preflight.** A line is red. Read its message, fix
it (`e` edits the configuration) and press `r` to check again.

**The report says "(report out of date)".** The QA was re-run after the
report was built. Run Products only.

**The sign-off disappeared after a rebuild.** The rebuild changed a number,
which voids the sign-off by design. Read the changes and sign off again.

**A report built with an earlier version reads differently.** The rules
for wording and ranges have been refined over time. Rebuild it with Products
only before comparing; if a client already has the earlier report, add a
revision note (section 4.10).

**The results map says "reload the page (F5)".** The dashboard or viewer
was restarted while the page was open. Press F5. A results map left open
across a software update keeps the old version until the dashboard (or
`insar viewer`) is restarted.

**The results map shows only the whole record, or says the points are out of
date.** Its point table predates the latest report or the date-window
feature. Run Products only, or
`python -m insar_pipeline.field_points <config.yaml>`.

**The results map does not list a project.** Its client KMZ has not been
built yet. Run Products only.

**An upload in the builder is refused.** The message says why: wrong format,
content that does not match the extension, a file over the limits, or a KML
with a DOCTYPE or ENTITY declaration. Export the layer again in one of the
accepted formats (section 4.2).

**A name is refused in the builder.** Use the plain name the client uses,
without underscores, colons, file extensions or placeholders.

**The doctor fails on memory.** See section 9.

### 8.5 Sending a problem to the custodian (`insar diagnose`)

When something goes wrong and you need the custodian (the maintainer) to look
at it, collect the project's records in one file. In the Ubuntu window, type
the command with the project's folder name (as `insar storage` lists it) in
place of `<project>`:

    insar diagnose <project>
    insar diagnose <project> --out DIR      # the zip goes to DIR instead

It writes a zip to `<projects root>/diagnostics/<project>_<YYYYmmdd-HHMMSS>.zip`
(or into DIR). Send that file to the custodian.

| In the zip | Not in the zip |
|---|---|
| `run_result.json` and `run_result_history.jsonl` | `context/` (the client's outlines) |
| `progress.jsonl` | the deliverables' maps, rasters and point files |
| `quality_assessment.json` | the report and the KMZ |
| the last three run logs from `logs/` | credentials (`~/.netrc`, `~/.cdsapirc`) |
| `deliverables/provenance.json` | the notifications log |
| `deliverables/config_snapshot.yaml`, if present | |
| the project's config file | |
| the JSON records in `deliverables/` (`client_report.json`, `gnss_tie.json`, the summary JSON files) | |
| the output of `insar --version` and of `insar doctor --quick` | |
| `MANIFEST.txt`, listing every file in the zip | |

Read `MANIFEST.txt` before you send the zip. Its text files write the home
folder as `~` and mask the Earthdata user name, but the config, the logs and
the JSON records still name the project and its processing box, so send the
zip only to the custodian.

### 8.6 Run logs and their level

Each run writes its log to `<project>/logs/<kind>_<time>.log`. The setting
`INSAR_LOG_LEVEL`, an environment variable, sets the run log's level:
`DEBUG` (the most detail), `INFO` (the default) or `WARNING` (warnings and
errors only).

Each run log is capped at 50 MB. When it is full it rolls over to `.1`, `.2`
and `.3`, so one run keeps four log files at most.

---

## 9. Disk space and memory

This chapter is for whoever looks after the computer: how much disk and
memory a run needs, how to free space, and how to remove a client's data
completely.

**Disk.** A phase-linked run peaks at about 300 GB. On WSL the real free
space is not what Linux reports: it is the WSL disk's headroom plus the free
space on the Windows drive that holds it. `insar doctor` and the preflight
show this real figure and warn below 350 GB. During a run, a watchdog warns
below 60 GB and stops the run cleanly below 20 GB, before the Windows drive
fills (which would corrupt files in every project).

**Freeing space.** Press `g` on the projects screen (or run `insar storage`)
for every project's size, state and how much each option would free:

- **Keep results only** (`k`, or `insar storage keep-results <name>`): deletes
  the downloads and processing files of a **finished** project. The report,
  deliverables, point products, logs, releases and sign-off stay. Afterwards
  only Products only is sure to work.
- **Delete project** (`d`, or `insar storage delete <name>`): deletes the whole
  project folder, after saving a small archive (the deliverables without
  files over 50 MB, except the report, KMZ, PDF and what went to the client;
  the validation; the logs; the run records and the configuration) to
  `_archive/<name>_<YYYYmmdd>/` in the projects folder. `--no-archive` keeps
  nothing. The configuration file in `configs/` is kept, so the project can
  be run again; delete it by hand to remove it (section 9.2).

Both show the plan first and do nothing until you confirm (in a terminal, add
`--apply` and type `yes`). Neither runs while a run is going. A project that
went to a client, or whose report is FINAL, also needs its name typed.

**Automatic cleanup.** When a project's FINAL client package is built, its
processing files are deleted and its results kept, as with Keep results only.
To keep everything, add `auto_keep_results: false` to its configuration.

**Cleanup from a terminal:**

    insar cleanup <config.yaml>                     # level 1, shows what it would delete
    insar cleanup <config.yaml> --level 2           # the large processing inputs, still a dry run
    insar cleanup <config.yaml> --level 2 --delete  # deletes; asks you to type 'yes'

After level 2 only Products only is sure to work. Cleanup never touches logs,
releases, the sign-off or the delivered products, and refuses while a run
holds the project.

**Memory.** `insar doctor` fails below 12 GB of memory and warns below 20 GB.
WSL receives half the computer's memory unless told otherwise. In Windows,
create or edit `C:\Users\<you>\.wslconfig`:

    [wsl2]
    memory=24GB
    swap=16GB

Use the computer's memory minus 6 to 8 GB (24GB on a 32 GB computer). Then
run `wsl --shutdown` in Windows PowerShell and reopen Ubuntu. The doctor
prints the exact lines for your computer. Worker counts are sized from the
machine's memory automatically; a value in the configuration overrides it.

### 9.1 Where the program writes

Every place the program writes to, in one table. `<root>` is the projects
folder (`~/insar_projects` unless `$INSAR_PROJECTS` or the settings file name
another); `<project>` is one project's folder in it. Sizes are for a full
phase-linked record of about ten years. "Cleanup" means `insar cleanup` at
the level named; "Keep results only" and "Delete project" are section 9's.
"By hand" means nothing in the program deletes it.

| Path | What's in it | Typical size | How it grows | What cleans it up |
|---|---|---|---|---|
| **Inside a project folder** | | | | |
| `<project>/` as a whole | everything below | peaks near 300 GB during a run; about 180 GB after it | a Products only rerun adds little | Keep results only; Delete project |
| `<project>/slc/` | downloaded satellite images (SAFE) | about 3 GB after the run | one per date during the download | each date's archive once it is coregistered; cleanup level 2 |
| `<project>/orbits/` | satellite orbit files | small | one per date | cleanup level 2 |
| `<project>/dem/` | the elevation model | small | once | Delete project only |
| `<project>/stack/` | the coregistered image stack (`coreg_secondarys/`), the merged images and interferograms (`merged/`), pairs set aside (`quarantine_unwrap_timeout/`) | about 60 GB | during coregistration and unwrapping | `merged/` and `coreg_secondarys/`: cleanup level 2 |
| `<project>/miaplpy/` | phase linking: the full-resolution image stack (`inputs/`) and the linked phases (`inverted/`) | about 100 GB | during phase linking | `inputs/` and `inverted/`: cleanup level 2 |
| `<project>/mintpy/` | the time series; `geo/` holds the final products, such as `geo/geo_velocity.h5` | about 11 GB | each time-series run | intermediate series and `pic/`: cleanup level 1; `geo/`: Delete project only |
| `<project>/mintpy/superseded/run_<stamp>/` | an earlier time-series run that a rerun replaced (`RETIRED.json` says what) | as large as the run it holds | one per rerun | cleanup level 1, on a finished project |
| `<project>/tropo_corrections/<MODEL>/` | weather delays, one folder per date (HRRR), or cached weather files (`weather_files/`, ERA5) | small to a few GB | one per date | `weather_files/`: cleanup level 2 |
| `<project>/interferograms/` | image pairs of the old classic route | depends on the route | per pair | cleanup level 1 (scratch) |
| `<project>/deliverables/` | the report, KMZ, PDF, rasters, point products, JSON records, the sign-off (`review.json`, `reviews/`), `provenance.json`, the results map's table (`viewer/`), client packages (`releases/<stamp>/`) | about 2 GB | each Products build; one folder per package | old `superseded_*` folders: cleanup level 1; the rest: Files page (name typed) or Delete project |
| `<project>/context/` | the client's outlines: project area, mine layers, report areas; the uploaded originals (`sources/`); what the builder reopens (`builder_layers.json`); the place name (`location.json`); cached map tiles (`basemap/`) | small | at creation; a report area saved from the results map adds one | Delete project only |
| `<project>/logs/` | one log per run (`<kind>_<time>.log`, rolled over at 50 MB); records kept from cleanup (`records/`); every deletion from the Files page (`deletions.jsonl`) | small; up to 200 MB a run | one log per run | Delete project only |
| `<project>/exports/maps/` | map exports and their `.json` option records | a few MB each | one per export | Files page; Delete project |
| `<project>/local_change/`, `feature_check/`, `analysis/vendor_compare/`, `validation/` | the optional layers and checks (section 6.10) and validation experiments | depends | when you run them | Files page; Delete project |
| `<project>/run.json`, `run_result.json`, `run_result_history.jsonl`, `progress.jsonl`, `quality_assessment.json`, `.run.lock`, `.run_children.json` | the run records | small | the history gains a line per run | Delete project only |
| **In the projects folder** | | | | |
| `<root>/configs/` | each project's configuration (`<name>.yaml`) and the copies kept when it is replaced (`<name>.yaml.bak_<time>`) | small | a copy per overwrite in the builder, or per report area saved from the results map | by hand |
| `<root>/gnss/` | GNSS station series, one folder per station, and the station catalogue; shared by every project | small | a station is added when a project needs it | by hand |
| `<root>/_archive/<name>_<YYYYmmdd>/` | the slim archive Delete project keeps | up to a few GB | one per deleted project | by hand |
| `<root>/run_history.jsonl` | one line per finished run of every project: the project's folder name, the kind of run, start and end times, status, stage, exit code, software version and code hash. No paths and no user names | small | a line per run | by hand |
| `<root>/diagnostics/` | the zips `insar diagnose` writes (section 8.5) | small | one per diagnose | by hand |
| **Outside the projects folder** | | | | |
| `~/.config/insar/settings.json` | the projects folder, firm, preparer, contact, logo setting, notification setting | tiny | when saved | by hand |
| `~/.config/insar/company_logo.png` | the company logo uploaded in the app | under 5 MB | replaced on upload | by hand (removing the logo in the app clears the setting, not the file) |
| `~/.insar_pipeline_cli.json` | the configurations used recently (up to 50 paths) | tiny | a path per project opened | Delete project takes the project's path off |
| `~/.insar_pipeline_notifications.log` | one line per finished run: time, `InSAR <project>: ...` and the message | small | a line per run | by hand |
| `~/.cache/insar_builder/` | the map builder page, and every uploaded file (`uploads/<id>/`) | up to 300 MB per upload | one folder per upload | uploads older than 2 days, each time the builder starts |
| `~/.cache/insar_viewer/` | a copy of the results map page | tiny | does not grow | replaced at each start |
| `~/.cache/insar_app/` | a copy of the app page, and the map tiles fetched for map exports (`tiles/`) | grows with every area exported | a tile per piece of map | nothing: tiles over 30 days old are fetched again, never deleted |
| `~/.cache/insar_pipeline/earthdata/` | the Earthdata session cookies for remote reads (readable by you only) | tiny | does not grow | by hand |
| `~/.cache/sentineleof/` | if present, the orbit download library's own cache (not written by this program) | small | | by hand |
| temporary folders | working files while a report or figure is built | small | | removed when the step ends |
| browser downloads | the results map's exports and chart downloads | small | one per download | by hand |
| **At install** | | | | |
| `~/.netrc`, `~/.cdsapirc` | the NASA Earthdata login and the Copernicus CDS token, readable by you only (mode 600) | tiny | | by hand |
| `~/.bashrc` | one line adding `~/.local/bin` to the PATH | | | by hand |
| `~/.local/bin/insar` | the `insar` command (a link into the software folder) | | | by hand |
| `~/insar_pipeline/` | the software itself; the Windows setup also leaves the release file it installed (`insar_pipeline_<version>_<hash>.tar.gz`) in the home folder | small | replaced by an update | by hand |
| `~/miniforge3/` and `~/miniforge3/envs/` | conda and the software environments (`insar`, `miaplpy`, `dolphin`, `burst2safe`, `raider-env`) | several GB | | by hand |
| `%USERPROFILE%\.wslconfig` (Windows) | WSL's memory setting | tiny | | by hand |
| Desktop and Start menu (Windows) | the **InSAR Pipeline** shortcuts; their launcher and icon in `%LOCALAPPDATA%\InSAR Pipeline\` | tiny | | by hand |

Everything inside Ubuntu (all but the last two rows) lives in the WSL disk
file on the Windows drive.

### 9.2 Removing a client's data

Use this when a client asks for its data to be removed, or a project must be
removed completely. It removes every copy the program keeps. Work through it
in order: step 1 reads records that step 2 deletes.

1. **Note what to look for.** The project's folder name (as `insar storage`
   lists it), its configuration file's name, the report title, the client's
   name and the site's name. Note the GNSS stations it used (`site` in
   `<project>/deliverables/gnss_tie.json`).
2. **Delete the project folder.** Make sure no run, cleanup or sign-off is
   going in it. Then, in an Ubuntu window:

       insar storage delete <name> --no-archive           # shows the plan only
       insar storage delete <name> --no-archive --apply   # deletes

   Type the project's name if asked (a delivered or FINAL project), then
   `yes`. This deletes the whole folder: `context/` with the client's
   outlines and originals, `deliverables/` and its packages, `exports/`,
   `logs/` and the run records. It also takes the project off
   `~/.insar_pipeline_cli.json`. It does **not** delete the configuration
   file, its `.bak_` copies, an earlier archive, a time-series folder kept
   outside the project folder (the plan says so when there is one), the
   shared GNSS folder, or anything outside the projects folder.
3. **Delete any earlier archive:** `rm -r <root>/_archive/<name>_*`.
4. **Delete the configuration and its copies:**
   `rm <root>/configs/<name>.yaml <root>/configs/<name>.yaml.bak_*`. On a
   developer's machine, also check the software folder's `configs/` and
   `context/` (`~/insar_pipeline/`) for the project's files.
5. **Remove its lines from the run history.** `<root>/run_history.jsonl`
   holds no paths or user names, only the folder name and the run's times and
   outcome. Remove the project's lines in a text editor, or:

       cd <root>
       grep -v '"project":"<name>"' run_history.jsonl > run_history.tmp && mv run_history.tmp run_history.jsonl

6. **Remove its lines from the notifications log** in the same way:

       grep -v 'InSAR <name>:' ~/.insar_pipeline_notifications.log > ~/notify.tmp && mv ~/notify.tmp ~/.insar_pipeline_notifications.log

7. **Delete the builder uploads.** They are removed 2 days after upload when
   the builder next starts. To remove them now, close the builder and run
   `rm -r ~/.cache/insar_builder/uploads`. Uploads are not named by project,
   so this removes every project's.
8. **Delete the cached map tiles:** `rm -r ~/.cache/insar_app/tiles`. They are
   public map images, but they show which places were exported.
9. **The GNSS folder.** The station data is public. Only the set of stations
   kept hints at where the site is. To remove that hint, delete
   `<root>/gnss/<SITE>/` for the stations from step 1 that no other project
   uses; they are downloaded again when needed.
10. **Delete any diagnose zips:** `rm <root>/diagnostics/<name>_*.zip`.
11. **Copies outside the program.** The program cannot find these; ask
    whoever had the project:
    - the results map's exports and chart downloads, in the browser's
      downloads folder;
    - copies made by hand: a KMZ or report e-mailed, a client package copied
      to a share or USB drive, files opened in Windows from
      `\\wsl.localhost\...`, Windows backups of those folders;
    - on a developer's machine, release files
      (`insar_pipeline_<version>_<hash>.tar.gz`, by default in the home
      folder). Current release files take only the public mine layers from
      `context/` and no project configurations; an older one may hold more.
      Check them, or delete them.
12. **Check.** Each of these should print nothing:

        grep -ril '<name>' ~/insar_projects ~/.cache ~/.config ~/.insar_pipeline_cli.json ~/.insar_pipeline_notifications.log
        grep -ril '<client or site name>' ~/insar_projects ~/.cache ~/.config

    Use your projects folder if it is not `~/insar_projects`.

Deleting in Ubuntu is permanent: WSL has no recycle bin. But the deleted
blocks stay inside the WSL disk file on the Windows drive until that file is
compacted. Compacting it is an IT step; ask IT if the removal must cover the
disk itself.

---

## 10. Settings you may need to change

Most projects never need a setting changed: the map builder writes the
production defaults. Change one only when this guide or the specialist says
so.

Press `e` on the project's live view or the preflight to edit the
configuration. Most settings should stay at the builder's defaults; these are
the ones this guide refers to.

| Setting | Effect |
|---|---|
| `end_date` | last image date; set it at least a week before today |
| `allow_missing_dates` | dates the run may proceed without, for example `['20231105']` |
| `stack_swaths` | the sub-swaths processed, for example `"2"` or `"1 2"` |
| `miaplpy_num_processors` | parallel workers in phase linking; lower it if phase linking stalls |
| `miaplpy_gpu` | phase linking on the graphics card: `auto` (default: the GPU when one works, else the CPU), `on` (GPU required), `off` (always the CPU). Same results as the CPU to rounding; about an hour instead of most of a day on a 445-date stack. Before linking, a short self-check compares it with MiaplPy on the CPU and stays on the CPU if they disagree |
| `report_areas` | the named areas the report gives figures for |
| `report_revision` | the reissue note (section 4.10) |
| `control_ground_label` | the word used for the comparison ground ("farmland" by default) |
| `miaplpy_skip` | pixels phase linking skips: `water` (default), `water_and_dense_woods`, or `none`; applies only when phase linking is (re)done |
| `landcover_label_wooded` | `false` turns off the tree-cover labelling |
| `local_change_sigma_km`, `local_change_threshold` | the local change layer (section 6.10): how wide the surroundings each step is read against (4 km), and the step quality needed to count (0.3) |
| `mask_wooded` | `false` keeps wooded cells in the maps and figures (by default they are left out: C-band sees the canopy, and woods read as rising against farmland) |
| `auto_keep_results` | `false` keeps all processing files after the package is built |
| `disp_client_report` | `true` builds a (never releasable) client report on the DISP-S1 route |

After changing a setting that affects only the report (names, report areas,
labels, the revision note), run Products only. Changes to processing settings
take effect only when the corresponding stage is run again.

---

## 11. Getting help

Most problems can be resolved with sections 8 and 9. Refer a project to the
designated InSAR specialist (the maintainer, Kyle Wire) when:

- the send rule is "hold for specialist";
- the same step fails twice in the same way after the recommended restart;
- the remedy would change processing settings (alignment network, re-linking,
  the datum gate, a broken image network);
- a GNSS station disagrees with the radar;
- a result looks implausible after checking imagery and dates.

To allow the issue to be assessed without further correspondence, include:

1. the project name and the path to its configuration file;
2. the stage that failed, or the send rule and its reasons;
3. the dashboard's **What now:** text, quoted in full;
4. the log file it names (`<project>/logs/<kind>_<time>.log`), or better, the
   zip from `insar diagnose <project>` (section 8.5), which holds it with the
   run records;
5. what you have already tried, and the result of each attempt.

Do not delete project folders, lock files or partial results while a referral
is open, unless this guide or the specialist directs you to.
