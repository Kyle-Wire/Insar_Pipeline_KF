# Installing the InSAR pipeline

**What this guide does.** It takes you from a Windows work laptop to a working
InSAR pipeline, one step at a time. You do not need to know Linux. Most of the
work is done by one setup program that you double-click.

**A release file is internal: do not distribute outside the operating
organisation.** It holds no client data. The `RELEASE.txt` file inside it says
the same, and `THIRD_PARTY.md` explains the licences. Do not give the software
to anyone outside the organisation.

## Before you start

You need all of these. Tick them off before you begin.

- [ ] **A Windows 10 or Windows 11 PC.** Windows 10 must be version 2004 or
  later. Ask IT if you are not sure.
- [ ] **Memory: 32 GB recommended, at least 24 GB.** The setup gives the
  pipeline the PC's memory minus 8 GB. The installation check fails below
  12 GB and warns below 20 GB, so a 16 GB PC does not pass it.
- [ ] **Free disk space on the C: drive** (or whichever drive holds Ubuntu,
  see below): about 10 GB for the program, plus space for projects. A
  full phase-linked run (the official route) peaks at about 300 GB, and
  the check warns when less than 350 GB is free. Quick DISP-S1 screening
  needs only a few GB.
- [ ] **Administrator rights, once.** Windows asks for them when it installs
  Ubuntu. If your account does not have them, ask IT to be there for that
  step (see section 6).
- [ ] **A good internet connection.** The setup downloads about 10 GB.
- [ ] **About 1 to 2 hours**, most of it waiting. Plug the laptop into power
  and stop it from going to sleep.
- [ ] **The release files from the maintainer (Kyle Wire).** Section 2 lists
  them.

**Words used in this guide**

- **Ubuntu** is a free version of Linux, a different operating system from
  Windows. The pipeline runs on Ubuntu.
- **WSL** (Windows Subsystem for Linux) is a part of Windows that runs Ubuntu
  inside Windows, like a program. You do not need a second computer.
- A **command window** (also called a **terminal**) is a window with plain
  text. Programs print messages in it, and you type answers or commands.
  The setup opens one. An **Ubuntu window** is a command window for Ubuntu;
  you open it from the Start menu by typing `Ubuntu`.
- **PowerShell** is the Windows command window. You need it only for one or
  two fixes in section 6. Open it from the Start menu by typing `PowerShell`.
- **"Type this and press Enter"** means: click in the command window, type the
  text exactly as shown (or copy and paste it), then press the Enter key.
  To paste into a command window, press Ctrl+V, or right-click in it.

## 1. Create two free data accounts (about 20 minutes)

The pipeline downloads satellite data and weather data from two public
services. Each needs a free account. Create both before you run the setup,
because the setup asks for them.

**NASA Earthdata** (satellite data, OPERA products and ionosphere maps):

1. Go to https://urs.earthdata.nasa.gov/users/new and create an account.
   Confirm your e-mail address when NASA asks.
2. Allow access to the Alaska Satellite Facility (ASF), which serves the
   Sentinel-1 data. Sign in once at https://search.asf.alaska.edu with your
   new Earthdata account and accept the terms when asked. (You can also do
   this on the Earthdata website: authorise "Alaska Satellite Facility Data
   Access" under Applications, Authorized Apps.)
3. Write down your Earthdata **user name** and **password**.

**Copernicus Climate Data Store (CDS)** (ERA5 weather data, used for projects
outside the continental US or before July 2016; US projects use NOAA HRRR
weather, which needs no account):

1. Go to https://cds.climate.copernicus.eu and create an account, then sign in.
2. Accept the ERA5 licence. Open the "ERA5 hourly data on pressure levels"
   dataset page,
   https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download,
   go to **Download data**, then **Terms of use**, and accept the
   **CC-BY licence**. This is a legal agreement, so the account holder must
   accept it.
3. Open your profile page and find your **API token** (a long string of
   letters and numbers). Keep the page open, or copy the token into a text
   file for now. You will paste it into the setup.

## 2. Put the release files in one folder (5 minutes)

The maintainer gives you four files. Put all four in **one** folder, for
example a new folder called `InSAR setup` in your Downloads folder. If they
came in a .zip file, right-click it and choose **Extract All** first: the
setup does not work from inside a zip.

| File | What it is |
|---|---|
| `insar_pipeline_<version>_<code>.tar.gz` | the program itself (one compressed file) |
| `insar_pipeline_<version>_<code>.tar.gz.sha256` | a check code that proves the file above arrived undamaged |
| `setup.ps1` | the setup program |
| `Setup InSAR Pipeline.cmd` | the file you double-click to start the setup |

`<version>` and `<code>` are the release's version and build code, for example
`insar_pipeline_1.4.0_3f9a2c1d.tar.gz`.

Check before you go on:

- All four files are there. Windows may hide the endings, so `setup.ps1` can
  show as "setup" and the `.cmd` file as "Setup InSAR Pipeline".
- The `.tar.gz` and `.sha256` names have not changed. If Windows added
  " (1)" or similar to a name, delete the extra copies and rename the file
  back. The setup refuses a release file with an unexpected name.
- There is only **one** release file in the folder. The setup uses the newest
  one it finds.

## 3. Run the setup (about 1 to 2 hours, mostly waiting)

You run the setup **twice**. The first run installs Ubuntu. The second run
installs the pipeline. Running it again later is safe: it skips what is
already done.

Every question the setup asks ends in `[y/N]`. Type `y` and press Enter for
Yes. Pressing Enter alone means No.

### 3.1 First run: install Ubuntu (about 15 minutes and a restart)

1. Double-click **Setup InSAR Pipeline.cmd**.
   A command window opens. You will see `InSAR pipeline setup` and then
   `== 1. Ubuntu (WSL2)`.
2. It says `!! Ubuntu is not installed in WSL yet.` and asks
   `Install Ubuntu now? [y/N]`. Type `y` and press Enter.
3. Windows asks "Do you want to allow this app to make changes to your
   device?". Choose **Yes**. This is the one time administrator rights are
   needed. A second window opens and installs Ubuntu. This takes 5 to 15
   minutes.
4. When it ends it shows
   `Done. Restart if Windows asked, then open Ubuntu once. Press Enter`.
   Press Enter. In the first window, press Enter again to close it.
5. If Windows asked you to restart, restart the PC now.
6. Open **Ubuntu** from the Start menu (type `Ubuntu` and click it). Ubuntu
   may also open by itself after the restart. The first time, it finishes
   installing and then asks you to create a Linux login:

   ```
   Enter new UNIX username:
   New password:
   Retype new password:
   ```

   Choose a short user name in lower-case letters with no spaces (for example
   your first name). Then choose a password. **Nothing appears on the screen
   while you type a password; this is normal.** Type it and press Enter.

   This user name and password are only for Ubuntu. They are separate from
   your Windows login. Write them down and keep them safe.
7. When you see a line ending in `$`, Ubuntu is ready. Close the Ubuntu
   window.

### 3.2 Second run: install the pipeline (about 1 hour)

1. Double-click **Setup InSAR Pipeline.cmd** again.
   You will see `OK  Ubuntu is ready (Linux user <your user name>)`.
2. **Memory.** You will see `== 2. Memory for WSL` and a line such as
   `OK  wrote memory=24GB, swap=16GB to C:\Users\<you>\.wslconfig (this PC has 32 GB)`.
   This tells Windows how much memory Ubuntu may use. It then asks
   `Nothing is running in Ubuntu now: restart WSL? [y/N]`. Nothing is running
   yet, so type `y` and press Enter. You will see `OK  WSL restarted`.
   (If you already had a `.wslconfig` with a memory line, it is kept.)
3. **Copying the program.** You will see `== 3. The pipeline in Ubuntu`,
   then `OK  checksum matches` (the release file arrived undamaged) and
   `OK  unpacked into ~/insar_pipeline`.
4. **The installer.** You will see
   `The installer runs now (about 30 minutes; it asks for the Earthdata login,`
   followed by a lot of text. It asks four questions:

   | It asks | What to type |
   |---|---|
   | `Earthdata username (Enter to skip):` | your NASA Earthdata user name, then Enter |
   | `Earthdata password:` | your Earthdata password, then Enter (nothing shows while you type) |
   | `CDS API token (Enter to skip):` | paste your CDS API token (Ctrl+V or right-click), then Enter |
   | `Projects folder [/home/<you>/insar_projects]:` | press Enter to accept the folder shown |

   The logins are saved inside Ubuntu, in your Linux home folder, readable
   only by your Linux user (Earthdata in `~/.netrc`, the CDS token in
   `~/.cdsapirc`). If you skip one, you can add it later (section 6).
5. **Downloads.** The installer then downloads and installs the software the
   pipeline uses, about 9 GB. Its progress lines start with `==`, for
   example `== 2/8 environment insar`. This takes 10 to 30 minutes or more,
   depending on your internet connection. Leave it running.
6. **The installation check.** The last part, `== 8/8 check`, runs
   `insar doctor` (section 4). You will see `… running the test suite` and
   then nothing for 10 to 30 minutes. This is normal: it is testing the
   program on your PC. Do not close the window.
7. The installer ends with
   `Done. Open a new terminal and type:  insar      (press ? in it for the user guide)`
   and the setup shows `OK  installed, and the installation check says Ready.`
   If the check found problems, it shows
   `!!  installed, but the installation check found problems` instead: look up
   the lines marked with a cross in section 6, fix them, and type
   `insar doctor` in the Ubuntu window to check again.
8. **The icon.** You will see
   `OK  'InSAR Pipeline' is on the desktop and in the Start menu`
   and `Open the InSAR Pipeline now? [y/N]`. Type `y` and press Enter to open
   it, or just press Enter. Then press Enter to close the setup window.

If the setup stops with a line starting `!!`, look it up in section 6.

## 4. Check that it worked (5 minutes)

**The installation check.** Scroll up in the setup window to the lines just
above `Done.`. A good result ends with:

```
  Ready.  Settings: /home/<you>/.config/insar/settings.json
```

Each line of the check starts with a mark:

- a green **✓** means the item is fine;
- a yellow **!** is a warning: you can still work, but read it (for example,
  too little disk space for a phase-linked run);
- a red **✗** is a problem. The line below it, starting with **→**, says how to
  fix it. If any red line is left, the check ends with
  `N problem(s) to fix.` instead of `Ready.`

You can run the check again at any time. Open Ubuntu from the Start menu,
type this and press Enter:

```
insar doctor
```

It takes 10 to 30 minutes, because it runs the test suite.
`insar doctor --quick` skips the test suite and takes about a minute.

**Your name on the reports.** Every report's footer says who prepared it, and
a report cannot be released without it. In an Ubuntu window, type this with
your own details and press Enter:

```
insar doctor --firm "Your Firm" --preparer "Your Name" --contact "you@yourfirm.com"
```

The contact address is also sent to OpenStreetMap as the contact for place
searches (section 10). Until you do this, the check shows a yellow warning
about the "report identity".

**The icon.** There is an **InSAR Pipeline** icon on the desktop and in the
Start menu.

## 5. Open the program

Double-click **InSAR Pipeline** on the desktop.

After a few seconds (longer the first time, while Ubuntu starts in the
background) the app window opens. It is a Microsoft Edge window with no tabs
or address bar; if Edge is not found, it opens in your normal web browser.
It shows your projects, live runs on a map, and results.

- Closing the window never stops a run. Double-click the icon again to
  reopen it.
- How to use the window: [USER_GUIDE.md, section 5.1 "The app window"](../USER_GUIDE.md#51-the-app-window).
- How to run a project from start to finish: [USER_GUIDE.md, section 4](../USER_GUIDE.md#4-running-a-project-step-by-step).

There is also a text version, the terminal dashboard. Open Ubuntu, type
`insar` and press Enter. Press `?` in it for the user guide (`USER_GUIDE.md`),
and `q` to quit. `insar app` opens the app window from an Ubuntu window. The
operator's checklist for a full run is `INSAR_RUNBOOK.md`.

**The two routes.**

- **Phase-linked:** the official results. 1 to 2 days and about 300 GB per
  area.
- **OPERA DISP-S1:** fast screening. A few hours; North America only. Not for
  official numbers, and its reports never pass the release gate.

## 6. If something goes wrong

First, run the setup again: it skips what is done and retries the rest. If the
same problem comes back, take a screenshot of the window (Windows key +
Shift + S) and send it to the maintainer.

**During the setup (Windows)**

| What you see | What it means | What to do |
|---|---|---|
| Windows asks for an administrator name and password, or the setup window closes straight after the administrator question | Your account does not have administrator rights, or you chose **No** | Run the setup again and choose **Yes**. Without administrator rights, ask IT to run the setup with you, or to run `wsl --install -d Ubuntu` in an administrator PowerShell window; then carry on from step 3.1, point 5. |
| Windows says "Windows protected your PC", or that running scripts is disabled | Windows or company policy blocks the downloaded files | Check the files came from the maintainer. Choose **More info**, then **Run anyway**. If your company blocks it, ask IT. |
| Installing Ubuntu fails with a message about virtualisation, "Virtual Machine Platform", or an error code such as 0x80370102 | The PC cannot run WSL2 yet: virtualisation is switched off, or WSL is blocked | Ask IT to enable WSL2 and virtualisation (in the PC's firmware settings) on this PC. Then run the setup again. |
| `!!  Ubuntu does not answer. Open 'Ubuntu' from the Start menu once (it finishes setting itself up), then run this again.` | Ubuntu is installed but has never been opened | Do step 3.1, points 6 and 7, then run the setup again. |
| `!!  Ubuntu runs as root: open 'Ubuntu' from the Start menu once and create your user, then run this again.` | Ubuntu was set up without a normal Linux user | Open Ubuntu from the Start menu. If it asks for a new user name, do step 3.1, point 6. If it does not, ask IT or the maintainer to create a Linux user and make it Ubuntu's default user. |
| `!!  this PC has N GB; the pipeline needs 12 GB for WSL and 32 GB in the PC is recommended` | The PC has too little memory | The installation will not pass the memory check. Talk to the maintainer before going on. |
| `!!  no release file found. Put insar_pipeline_<version>.tar.gz (and its .sha256) beside this` | The `.tar.gz` file is not in the same folder as the setup | Put all four files in one folder (section 2) and run the setup again. (The rest of the message, about cloning, is for the maintainer: section 11.) |
| `!!  unexpected release file name: ...` | The release file was renamed, for example with " (1)" added | Rename it back to its original name (section 2). |
| `!!  the release file does not match its .sha256 (...): copy it again` | The release file was damaged or cut short while it was copied or downloaded | Get the file again from the maintainer and replace it. Do not use the damaged file. |
| `!!  no .sha256 beside the release file: its integrity is not checked` | The `.sha256` file is missing (the setup carries on) | Better: stop (close the window), get the `.sha256` file from the maintainer and run the setup again. |
| `!!  unpacking failed` | The release file could not be unpacked; often the disk is full | Free space on the C: drive and run the setup again. |
| `!!  the installer stopped with an error (above). Fix it and run this setup again.` | Something in the installer failed, often a download that was cut off | Check the internet connection and run the setup again. When it asks `Run its installer again ...? [y/N]`, type `y`. If it fails twice in the same place, send the maintainer a screenshot of the text above that line. |
| `!!  cannot find install/windows in ~/insar_pipeline` | The program is not inside Ubuntu, so the icon cannot be made | Run the setup from the folder with all four release files. |

**In the installation check (`insar doctor`)**

To redo a login: open Ubuntu and type the first line below for Earthdata, or
the second for CDS (each deletes the saved login), then the third, which runs
the installer again. It asks for that login again; press Enter at the
projects folder question.

```
rm ~/.netrc
rm ~/.cdsapirc
bash ~/insar_pipeline/install/install.sh
```

| What you see | What it means | What to do |
|---|---|---|
| ✗ `NASA Earthdata login in ~/.netrc` | No Earthdata login was entered | Run the installer again (the third line above) and enter it. |
| ✗ `Earthdata login fails: ...` with `→ fix the user name / password in ~/.netrc` | The user name or password is wrong | Redo the Earthdata login (above). |
| ✗ `Earthdata login fails: The Earthdata login for '...' works, but the account has not authorised ASF data access ...` | Step 1, NASA point 2 was not done | Sign in once at https://search.asf.alaska.edu and accept the terms, then run `insar doctor` again. |
| ! `Earthdata login not checked: the Earthdata / ASF servers did not answer; try again later` | NASA's servers are busy or down; your login may be fine | Try again later. Do not change the login. |
| ! `Copernicus CDS key in ~/.cdsapirc (ERA5 weather)` | No CDS token was entered | Needed for projects outside the continental US or before July 2016. Run the installer again and paste the token. |
| ✗ `ERA5 access: The CDS token works but the ERA5 licence 'CC-BY licence' (cc-by) has not been accepted on this account ...` | Step 1, CDS point 2 was not done | Accept the CC-BY licence on the ERA5 page (step 1), then run `insar doctor` again. |
| ✗ `ERA5 access: CDS rejected the token in ...` | The token was mistyped or is old | Copy the API token again from your CDS profile page and redo the CDS login (above). |
| ✗ memory ... `- below 12 GB` | Ubuntu has too little memory | If the setup's WSL restart was skipped: close all Ubuntu windows, open PowerShell, type `wsl --shutdown` and press Enter, then run `insar doctor` again. Otherwise the PC needs more memory: talk to the maintainer. |
| ! memory ... `- below the 20 GB the pipeline was measured on` | Ubuntu has enough memory to work, but less than the pipeline was tested with | Runs may be slower. A 32 GB PC is recommended. |
| ! `room for a phase-linked run (~300 GB peak; N GB available)` | Less than 350 GB is free | Fine for DISP-S1 screening. For phase-linked runs, free space on the Windows drive, or move the projects folder (section 8). |
| ✗ `release files: N of M shipped files differ from the release ...` | Program files were changed or damaged after the release was built | Reinstall the release (section 7). |
| ✗ `test suite: ...` | The program does not compute what it should on this PC | Run the installer again (`bash ~/insar_pipeline/install/install.sh`), then `insar doctor`. If it still fails, send the maintainer the line. |
| ✗ `test suite: did not finish in 1800 s` | The tests took longer than 30 minutes | Close other programs and run `insar doctor` again. If it happens again, tell the maintainer. |
| ! `report identity: (firm not set) ...` | Your name and firm are not set | Run the `insar doctor --firm ...` command in section 4. |
| any other ✗ line | Part of the installation is missing | Follow the line starting with **→** under it. If that does not fix it, send the maintainer the full output. |

**Using the program**

| What you see | What it means | What to do |
|---|---|---|
| Double-clicking **InSAR Pipeline** does nothing | The window can take up to a minute the first time; or the program cannot start | Wait a minute. Then open Ubuntu, type `insar app` and press Enter: any error is printed there. Send it to the maintainer. |
| The icon is missing | It was deleted, or the setup stopped before making it | Run the setup again. Answer `N` (just press Enter) when it asks to run the installer again; it then remakes the icon. |
| `insar: command not found` in an Ubuntu window | The window was opened before the installation finished | Close it and open a new Ubuntu window. |

## 7. Updating to a new release

The setup does **not** install a new release over an existing one: when the
pipeline is already installed, it only offers to rerun the installer of the
copy you have. Update in an Ubuntu window instead (about 15 minutes, plus the
installation check).

1. Make sure no run is going.
2. Copy the new release file and its `.sha256` into your Ubuntu home folder.
   In Windows File Explorer, type `\\wsl.localhost\Ubuntu\home\<your Ubuntu user name>`
   into the address bar and press Enter, then paste the two files there.
3. Open Ubuntu from the Start menu. Type each line below and press Enter after
   each. Replace `<version>_<code>` with the new file's name.

   ```
   cd ~
   sha256sum -c insar_pipeline_<version>_<code>.tar.gz.sha256
   ```

   You will see the file name followed by `OK`. If it says `FAILED`, stop and
   get the file again.

   ```
   mv insar_pipeline "insar_pipeline_old_$(date +%Y%m%d)"
   tar -xzf insar_pipeline_<version>_<code>.tar.gz
   bash insar_pipeline/install/install.sh --no-credentials
   insar --version
   ```

   The first line moves the old copy aside (it is your way back). The last
   line shows the new version.

Unpacking over the old folder is not safe: files removed or renamed in the
new release would stay behind and keep running.

The installer repoints the `insar` command and the Python install at the new
folder, and moves the patched MiaplPy's pip lock to it. `--no-credentials`
skips the login questions. Your projects, settings
(`~/.config/insar/settings.json`) and logins are not touched: the projects
folder stays the folder already in use (`~/insar_projects`, or the one set
with `insar doctor --projects`). The desktop icon keeps working.

It keeps each existing software environment, but checks it against the new
release's pins in `install/envs/`. When one differs, it lists the differences
and the command that rebuilds it, for example:

    conda env remove -n insar && bash ~/insar_pipeline/install/install.sh

A snaphu 2.0.7 that still contains the CS2 module is rebuilt without it; the
official route's unwrapping is unchanged by this (checked bit for bit, see
`THIRD_PARTY.md`).

**To go back** to the old release, in an Ubuntu window:

    cd ~
    mv insar_pipeline insar_pipeline_bad && mv insar_pipeline_old_<date> insar_pipeline
    bash insar_pipeline/install/install.sh --no-credentials

Delete the old folder once the new release has run a project.

## 8. Moving the projects folder

Projects are kept in `~/insar_projects` inside Ubuntu unless you chose another
folder during the installation. To use a different folder, for example one
with more space, type this in an Ubuntu window (with the folder's real path)
and press Enter:

    insar doctor --projects /path/to/new/folder

This saves the new folder in your settings and runs the installation check,
which then shows the new folder's free space. It does **not** move projects
you already have; ask the maintainer before moving existing projects. Running
the installer again (an update, for example) keeps the folder already in use:
it never moves the projects folder back to `~/insar_projects`. (A folder set
in the `INSAR_PROJECTS` environment variable overrides the saved setting.)

## 9. Removing the program

To remove only the icon: delete **InSAR Pipeline** from the desktop and the
Start menu, and the folder `%LOCALAPPDATA%\InSAR Pipeline` (type it into the
File Explorer address bar).

To remove the pipeline itself, ask the maintainer. The program, its software
environments, your logins and (by default) all your projects live inside
Ubuntu, so removing it can delete client data. Follow
[USER_GUIDE.md, section 9.2 "Removing a client's data"](../USER_GUIDE.md#92-removing-a-clients-data)
first, and keep copies of anything you need.

## 10. What leaves this computer

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
its approval status, is `EXTERNAL_SERVICES.md`. The personal and client
information the program keeps is described in `PRIVACY.md`.

## 11. For the maintainer and advanced users

### What the setup does

`install/windows/setup.ps1` (started by `Setup InSAR Pipeline.cmd`, which
passes its arguments on) does four things, each skipped or asked about when
already done:

1. Ubuntu in WSL2: `wsl --install -d Ubuntu`, elevated once. It then stops;
   Ubuntu must be opened once to create the Linux user, and the setup run again.
2. WSL's memory: writes `memory=<PC's RAM minus 8 GB, at least 12>GB` and
   `swap=16GB` to `%USERPROFILE%\.wslconfig` under `[wsl2]` (a copy of an
   existing file is kept as `.wslconfig.bak-<date>`), and offers
   `wsl --shutdown`. A `.wslconfig` that already sets `memory` is left alone.
3. The pipeline: copies the newest `insar_pipeline_*.tar.gz` beside it (and its
   `.sha256`) into the Linux home, checks it with `sha256sum -c`, unpacks it to
   `~/insar_pipeline` and runs `install/install.sh`. If `~/insar_pipeline`
   already holds an installer, nothing is copied or unpacked; if the `insar`
   command is also installed, it only offers to rerun that installer.
4. The icon: copies `InSAR Pipeline.vbs` and `insar.ico` to
   `%LOCALAPPDATA%\InSAR Pipeline` with a `distro.txt` naming the
   distribution, and makes desktop and Start-menu shortcuts. The shortcut runs
   `insar app` in Ubuntu with no console window.

Options: `-Release <file.tar.gz>` (a release elsewhere), `-Distro <name>`
(default `Ubuntu`), `-IconOnly` (step 4 only). On a PC installed from git, run
this from `\\wsl.localhost\Ubuntu\home\<you>\insar_pipeline` to add just the
icon:

    powershell -ExecutionPolicy Bypass -File install\windows\setup.ps1 -IconOnly

### Installing by hand in Ubuntu

**Ubuntu.** In PowerShell as administrator: `wsl --install -d Ubuntu`.
Restart when asked, open **Ubuntu** from the Start menu and choose a user name
and password.

**Memory.** WSL gets half the PC's memory unless told otherwise. The doctor
fails below 12 GB for WSL and warns below 20 GB, so a 32 GB PC is recommended.
Create or edit `C:\Users\<your Windows user>\.wslconfig`:

    [wsl2]
    memory=24GB
    swap=16GB

Use the PC's memory minus 6 to 8 GB (24GB on a 32 GB PC). Then run
`wsl --shutdown` in PowerShell and reopen Ubuntu. `insar doctor` checks this
and prints the exact lines for your PC.

**Disk.** A phase-linked run peaks at about 300 GB and lives on the drive that
holds WSL's disk (usually C:). The real space available is the WSL disk's
headroom plus that drive's free space, not what Linux reports; the doctor and
the dashboard show the real figure.

**From a release file.** Copy `insar_pipeline_<version>_<code>.tar.gz` and its
`.sha256` into the Ubuntu home folder (`\\wsl.localhost\Ubuntu\home\<your Ubuntu username>`
in Windows Explorer), then in the Ubuntu window:

    cd ~
    sha256sum -c insar_pipeline_*.tar.gz.sha256
    tar -xzf insar_pipeline_*.tar.gz
    bash insar_pipeline/install/install.sh

After installing, `insar doctor` checks every shipped file against the
release's `MANIFEST.sha256`.

**What `install/install.sh` does** (each step skipped when already done; safe
to re-run; `--no-credentials` skips the login and projects-folder questions):

1. Miniforge (conda) into `~/miniforge3`, if no conda is found;
2. five conda environments from `install/envs/`, at the exact versions this
   was validated with (conda-forge only; about 9 GB, 10 to 30 minutes):
   `insar`, `miaplpy`, `burst2safe` (the downloads), `raider-env` (NOAA HRRR
   weather delays through RAiDER) and `dolphin` (phase linking for the NISAR
   phase-linked route). An existing environment is kept but checked against
   the pins;
3. this pipeline into the `insar` environment (`pip install -e`);
4. snaphu 2.0.7 built without its noncommercial CS2 module, into `miaplpy`
   (`install/build_snaphu.sh`);
5. the patched MiaplPy (`patches/miaplpy/README.md`), with its pip lock
   pointed at this folder;
6. the logins: NASA Earthdata into `~/.netrc`, the Copernicus CDS token into
   `~/.cdsapirc` (both mode 600). It asks only for a login not already saved;
7. the projects folder (Enter accepts `~/insar_projects`, or on a re-run the
   folder already in use) and the `insar` command in `~/.local/bin`;
8. `insar doctor`, which should end in **Ready.** The installer does not stop
   when the doctor reports problems.

The folder must be called `insar_pipeline`: the installer refuses to run from
a folder with any other name.

### Installing from a git clone

Make the SSH key in the Ubuntu window, not in PowerShell: git in Ubuntu cannot
use a key made on the Windows side. Add the printed public key to your account
on the site that hosts the repository (the maintainer gives you access), then
clone and install:

    ssh-keygen -t ed25519
    cat ~/.ssh/id_ed25519.pub
    git clone <this repository's SSH address> ~/insar_pipeline
    bash ~/insar_pipeline/install/install.sh

To update later: `cd ~/insar_pipeline && git pull`, then re-run the installer
(finished steps are skipped). A git checkout has no `MANIFEST.sha256`, so the
doctor's release-files line says it is a development checkout. Add the Windows
icon with `-IconOnly` (above).

### Making a release (Kyle)

Bump `version` in `pyproject.toml`, run the tests, then:

    bash ~/insar_pipeline/install/make_release.sh [output folder]

It writes `insar_pipeline_<version>_<code hash>.tar.gz` (default: your home
folder), and beside it `<same name>.tar.gz.sha256` (checked with
`sha256sum -c`), `setup.ps1` and `Setup InSAR Pipeline.cmd`: the four files a
coworker needs (section 2). The tarball has one top folder, `insar_pipeline/`,
and is built from a whitelist: the code, `scripts/`, `install/` (with the
Windows setup and the icon's launcher), the MiaplPy patch, `tests/`,
`context/`, the documents, and a `RELEASE.txt` naming the build. It never
ships personal settings folders, Windows `:Zone.Identifier` files, the stray root
`config.yaml` and `project_zone.geojson`, `backups/`, `review_*/`, the
archived configs in `configs/archive/` (only the archived topsApp runbook
ships), `report_assets/`, caches, or anything from the projects folder, and
refuses to build if any of those slips in.

It carries no client work (2 Nov 2026). No project config ships: projects keep
theirs in the projects folder's `configs/`. The machine's local-only tests
(`tests/test_local_*.py`, which read client projects), the golden client
numbers and any client report areas in `context/` are left out even when the
release is built on the developer's machine; `context/` ships only the public
Indiana mine layers (IGWS and Indiana DNR; `context/SOURCE.txt`), and the
build refuses to finish if anything else is there. Every release is still
internal: do not distribute outside the operating organisation. `RELEASE.txt`
carries that line, and the build prints it.

Each release also carries `SBOM.cdx.json`, a CycloneDX list of every pinned
package in every environment (`scripts/make_sbom.py`), and `MANIFEST.sha256`,
the sha256 of every shipped file, which `insar doctor` checks after an
install. The engineering log (HANDOFF.md) does not ship: it stays on the
developer's machine.

### Removing everything

`wsl --unregister Ubuntu` in PowerShell deletes the whole Ubuntu system:
the pipeline, its environments, the logins, the settings and every project
inside it. It cannot be undone. Copy out anything to keep first, and follow
USER_GUIDE.md section 9.2 for client data. Then remove the icon as in
section 9.

### Notes

- The three comparisons against `opera-utils` in the test suite show as
  SKIPPED (`pytest -rs` lists them): `opera-utils` is not installed in the
  `insar` environment.
- snaphu is built with `-DNO_CS2`. A config asking for `INITMETHOD MCF` stops
  with "program not compiled with cs2 MCF solver module". The official route
  uses MST. The classic route's `isce` unwrap engine (comparison only) uses
  the snaphu built into ISCE2, which contains CS2; keep
  `stack_unw_full_cost: true` there.
- To rebuild snaphu by hand: `bash install/build_snaphu.sh miaplpy`
  (`SNAPHU_TARBALL=<file>` uses a local copy of the source, `SNAPHU_DEST=<dir>`
  installs elsewhere for a test).
- The doctor's test suite takes about 9 minutes on a fast PC (about 1,960
  tests); its time limit is 30 minutes.
