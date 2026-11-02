# Personal and client information

What personal and client-identifying information the InSAR pipeline keeps,
where, who sees it, what leaves the machine, and how it is deleted. Built from
the code on 2 Nov 2026. `<project>` is a project's folder; `<root>` is the
projects folder (`~/insar_projects` by default). Every place the program
writes is listed in `USER_GUIDE.md` section 9.1.

| Information | Where it is kept | Who sees it | What leaves the machine |
|---|---|---|---|
| Preparer's name, firm, contact | `~/.config/insar/settings.json` (set with `insar doctor --firm ... --preparer ... --contact ...` or the app's **Company**) | the client: the report footer, the map exports' title block, and `provenance.json` in the package | the firm and contact, in the User-Agent of every request to OpenStreetMap Nominatim (the builder's place search and the report's place name) and to the USGS National Map (the report's base map). Map-export tile requests carry only the software version |
| Company logo | `~/.config/insar/company_logo.png` | the client, on map exports | nothing |
| Reviewer's and specialist's names, sign-off notes | `<project>/deliverables/review.json`; earlier sign-offs in `deliverables/reviews/` | the client: the package's `technical/` folder holds the review | nothing |
| Project names and site locations | the project's folder name and configuration (`<root>/configs/`), its outlines, `<root>/run_history.jsonl` (folder name only), `~/.insar_pipeline_notifications.log`, `~/.insar_pipeline_cli.json` (configuration paths), `<root>/_archive/`, `<root>/diagnostics/` | anyone with the Ubuntu account; the client (the report); the custodian (a diagnose zip) | the site's location, in some form, to every service in `EXTERNAL_SERVICES.md` (the satellite archive, weather, elevation, land cover, water map, place name and map tiles). No service receives a client's name, a report or report numbers. A notification command, if one is set in the settings, sends the run's end message where it points |
| Client-supplied outlines and their original files | `<project>/context/` (the originals in `context/sources/`); while the builder is in use, `~/.cache/insar_builder/uploads/` | anyone with the Ubuntu account; the client sees the outlines in the report and KMZ | the outline or processing box to the satellite archive search (ASF, NASA CMR) |
| `provenance.json` (what produced a result) | `<project>/deliverables/provenance.json` | the client, in the package's `technical/` folder | nothing |
| Earthdata and Copernicus CDS logins | `~/.netrc` and `~/.cdsapirc`, outside the software folder, readable by you only (mode 600); Earthdata session cookies in `~/.cache/insar_pipeline/earthdata/` (readable by you only) | you | only to NASA Earthdata and the Copernicus CDS |

**provenance.json** names the operator by the preparer's name from the
settings, not the computer login or the computer's name, and writes the home
folder as `~` in every path. It also holds the processing box and the dates.

**Exceptions found in the code** (not yet changed):

- `deliverables/client_report.json` records the project folder's full path
  (`project`), which includes the login name. It goes to the client in the
  package's `technical/` folder.
- `insar doctor` and the preflight print the Earthdata user name when they
  check the login (`NASA Earthdata login for '<user>' works`), so it can
  appear in the output a diagnose zip holds. The password and the CDS token
  are never printed.
- Run logs and the run record `run.json` can hold paths that include the
  login name.

**How long it is kept, and how it is deleted.** Remove a client's data with
the procedure in `USER_GUIDE.md` section 9.2 ("Removing a client's data"). The
firm, preparer, contact and logo are removed by editing or deleting
`~/.config/insar/settings.json` and `company_logo.png`; the logins by deleting
`~/.netrc`, `~/.cdsapirc` and `~/.cache/insar_pipeline/earthdata/`. Copies
already sent (a package, a KMZ, a diagnose zip) are outside the program's
reach.

Retention: project data is kept until the project is deleted; nothing is
deleted automatically except builder uploads after two days (removed the next
time the builder starts). Processing files, which hold public satellite data,
are also removed automatically once a FINAL package is built.

This inventory has not yet had a privacy review.
