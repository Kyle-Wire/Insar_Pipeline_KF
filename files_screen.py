"""The dashboard's file browser: projects, deliverables and logs without
leaving the terminal.

  arrows / Enter   move, open folders
  o                open the file (or folder) with the computer's own program;
                   a file over ~200 MB is shown in its folder instead (see below)
  r                show it in the file manager (Windows Explorer on WSL)
  c                copy its path (the Windows path too, on WSL)
  t                move it to the trash (<projects>/.trash, recoverable)
  e                empty the trash (asks, shows the size)
  h                back to the projects folder;  u  up one level
  Esc              back to the dashboard

The preview on the right: size and date for everything; the text of small
text files (JSON pretty-printed); the size, grid, projection and value range
of a GeoTIFF; the datasets of an HDF5 file; the total size of a folder;
what a delivered layer means (the release package's own one-line meanings);
and, on a DISP-S1 screening without a client report, why there is none.
Nothing is deleted outright: "trash" moves files into <projects>/.trash/<date>,
and only "empty the trash" frees the space.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import DirectoryTree, Footer, Header, Static

from . import settings

TEXT = {".json", ".yaml", ".yml", ".txt", ".log", ".csv", ".md", ".cfg", ".py", ".geojson", ".xml"}

# `o` hands a file to Windows' own program for it. Site B's
# measurement_points_egms.csv is 8.1 GB / 2,584,187 points: Excel stops at
# 1,048,576 rows (it warns once, then a coworker works on 40% of the table),
# and Explorer stalls on 8-14 GB. Above this size `o` shows the file in its
# folder and says what to open it with instead - the same 200 MB the release
# package uses to list a technical file instead of copying it.
BIG_FILE_BYTES = 200e6
EXCEL_MAX_ROWS = 1_048_576
BIG_FILE_ADVICE = {
    ".csv": (f"Excel loads only the first {EXCEL_MAX_ROWS:,} rows of a table this size. For one area use the "
             "dashboard's Results (v) -> Export (per-area Excel); for every point open "
             "points_attributes.gpkg in QGIS."),
    ".gpkg": "Open it in QGIS (Layer > Add Layer > Add Vector Layer); Windows has no program for it.",
    ".parquet": "Open it in QGIS 3.28+ or with Python (geopandas.read_parquet); Excel cannot read it.",
    ".kmz": "Google Earth slows to a stop on a KMZ this size; client_points.kmz is the one to open.",
    ".h5": "HDF5 has no Windows program; the preview here lists its datasets.",
}
SCREENING_NOTE = "SCREENING_NOT_FOR_CLIENTS.txt"     # pipeline.SCREENING_NOTE


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(p: Path, budget_s: float = 8.0) -> tuple[int, int, bool]:
    """(bytes, files, complete) - stops after budget_s on huge trees."""
    total = files = 0
    t0 = time.time()
    for f in p.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
                files += 1
        except OSError:
            pass
        if time.time() - t0 > budget_s:
            return total, files, False
    return total, files, True


def windows_path(p: Path) -> str | None:
    if shutil.which("wslpath"):
        r = subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True)
        return r.stdout.strip() or None
    return None


def open_default(p: Path) -> str:
    """Open with the computer's own program for that file type."""
    win = windows_path(p)
    if win and shutil.which("explorer.exe"):
        subprocess.Popen(["explorer.exe", win], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return win
    if shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(p)


def open_advice(p: Path) -> str | None:
    """Why `o` should not hand this file to the computer's own program, and
    what to use instead; None for a file it can open (a folder, a small file)."""
    try:
        if p.is_dir() or p.stat().st_size < BIG_FILE_BYTES:
            return None
        size = p.stat().st_size
    except OSError:
        return None
    return (f"{p.name} is {human(size)}. " + BIG_FILE_ADVICE.get(p.suffix.lower(), "Opening a file this size can "
            "stall Windows; open it in the program meant for it.") + " Showing it in its folder instead.")


def layer_meaning(p: Path) -> str | None:
    """A delivered layer's one-line meaning, as the release package's README
    gives it (package.DATA_FILES / TECH_FILES: quality_unwrapping_consistency,
    quality_too_fast_to_measure, quality_position_uncertainty_m,
    disturbed_ground_runs, dem_difference_m, the point products...)."""
    try:
        from . import package
        rules = tuple(getattr(package, "DATA_FILES", ())) + tuple(getattr(package, "TECH_FILES", ()))
    except Exception:                                         # noqa: BLE001
        return None
    return next((meaning for rx, meaning in rules if rx.search(p.name)), None)


def screening_note(p: Path) -> str | None:
    """On a DISP-S1 screening (the _disp_screening folder, or a DISP chain
    record) with no client report: say so, and where the reason is written,
    rather than leave a coworker hunting for the report."""
    P = p.parent if p.name == "deliverables" else p
    if not ((P / "mintpy" / "disp_chain.json").exists() or P.name.endswith("_disp_screening")):
        return None
    if (P / "deliverables" / "client_report.json").exists():
        return None
    try:
        from .pipeline import SCREENING_NOTE as name
    except Exception:                                         # noqa: BLE001
        name = SCREENING_NOTE
    return f"screening: no client report (see deliverables/{name})"


def reveal(p: Path) -> str:
    """Show it selected in the file manager."""
    win = windows_path(p)
    if win and shutil.which("explorer.exe"):
        subprocess.Popen(["explorer.exe", f"/select,{win}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return win
    return open_default(p if p.is_dir() else p.parent)


def trash_dir() -> Path:
    return settings.projects_root() / ".trash"


def move_to_trash(p: Path) -> Path:
    dest = trash_dir() / datetime.now().strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / p.name
    shutil.move(str(p), str(target))
    return target


class FilesScreen(Screen):
    BINDINGS = [("o", "open", "Open"), ("r", "reveal", "Show in folder"), ("c", "copy", "Copy path"),
                ("t", "trash", "Trash"), ("e", "empty", "Empty trash"), ("u", "up", "Up"),
                ("h", "home", "Projects"), ("escape", "back", "Back")]

    def __init__(self, root: Path | None = None):
        super().__init__()
        self.root = Path(root or settings.projects_root())
        self.current: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="fpath")
        with Horizontal(id="fbody"):
            yield DirectoryTree(str(self.root), id="ftree")
            with VerticalScroll(id="fprev"):
                yield Static(id="finfo")
        yield Footer()

    def on_mount(self) -> None:
        self._title()
        self.query_one(DirectoryTree).focus()

    def _title(self) -> None:
        t = Text()
        t.append(" Files ", style="bold reverse")
        t.append(f"  {self.root}", style="dim")
        self.query_one("#fpath", Static).update(t)

    def _set_root(self, p: Path) -> None:
        self.root = p
        tree = self.query_one(DirectoryTree)
        tree.path = str(p)
        tree.reload()
        self._title()

    @on(DirectoryTree.NodeHighlighted)
    def _highlight(self, ev) -> None:
        data = getattr(ev.node, "data", None)
        path = getattr(data, "path", None)
        if path is not None:
            self.current = Path(path)
            self._preview(self.current)

    @work(thread=True, exclusive=True, group="preview")
    def _preview(self, p: Path) -> None:
        self.app.call_from_thread(self.query_one("#finfo", Static).update, Text("…", style="dim"))
        self.app.call_from_thread(self.query_one("#finfo", Static).update, describe(p))

    # ---- actions
    def action_open(self) -> None:
        if not self.current:
            return
        why = open_advice(self.current)
        if why:
            self.notify(f"{why} {reveal(self.current)}", severity="warning", timeout=15)
            return
        self.notify(f"Opening {open_default(self.current)}")

    def action_reveal(self) -> None:
        if self.current:
            self.notify(f"Showing {reveal(self.current)}")

    def action_copy(self) -> None:
        if not self.current:
            return
        text = windows_path(self.current) or str(self.current)
        self.app.copy_to_clipboard(text)
        if shutil.which("clip.exe"):
            subprocess.run(["clip.exe"], input=text.encode("utf-16le"), check=False)
        self.notify(f"Copied {text}")

    def action_up(self) -> None:
        if self.root.parent != self.root:
            self._set_root(self.root.parent)

    def action_home(self) -> None:
        self._set_root(settings.projects_root())

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_trash(self) -> None:
        from .dashboard import Confirm
        p = self.current
        if p is None or not p.exists():
            return
        if trash_dir() in p.parents or p == trash_dir():
            self.notify("That is already in the trash; use e to empty it.", severity="warning")
            return
        if p == settings.projects_root() or p in Path(settings.PKG).parents or p == settings.PKG:
            self.notify("Refusing to trash the projects folder or the software itself.", severity="error")
            return

        def go(yes):
            if yes:
                try:
                    dest = move_to_trash(p)
                    self.notify(f"Moved to trash: {dest}")
                    self.query_one(DirectoryTree).reload()
                except Exception as exc:                     # noqa: BLE001
                    self.notify(f"Could not move it: {exc}", severity="error")
        kind = "folder" if p.is_dir() else "file"
        self.app.push_screen(Confirm(f"Move this {kind} to the trash?\n\n{p}\n\nIt stays recoverable in "
                                     f"{trash_dir()} until the trash is emptied.", "Move to trash"), go)

    def action_empty(self) -> None:
        from .dashboard import Confirm
        t = trash_dir()
        if not t.exists() or not any(t.iterdir()):
            self.notify("The trash is empty.")
            return
        size, n, done = dir_size(t, 20.0)

        def go(yes):
            if yes:
                shutil.rmtree(t)
                self.notify(f"Trash emptied: {human(size)} freed.")
                self.query_one(DirectoryTree).reload()
        self.app.push_screen(Confirm(f"Permanently delete everything in the trash?\n\n{t}\n"
                                     f"{n:,} files, {human(size)}{'' if done else ' or more'}. This cannot be undone.",
                                     "Delete permanently"), go)


def describe(p: Path) -> Text:
    """The preview for one path."""
    t = Text()
    try:
        st = p.stat()
    except OSError as exc:
        return Text(str(exc), style="red")
    t.append(f"{p.name}\n", style="bold")
    t.append(f"{p}\n", style="dim")
    win = windows_path(p)
    if win:
        t.append(f"{win}\n", style="dim")
    t.append(f"modified {datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}\n\n")
    if p.is_dir():
        note = screening_note(p)
        if note:
            t.append(note + "\n\n", style="bold yellow")
        size, n, done = dir_size(p)
        t.append(f"{n:,} files, {human(size)}{'' if done else ' so far (large folder - stopped counting)'}\n\n", style="bold")
        try:
            kids = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[:60]
            for k in kids:
                t.append(("▸ " if k.is_dir() else "  ") + k.name + "\n", style="cyan" if k.is_dir() else "")
        except OSError:
            pass
        return t
    t.append(f"{human(st.st_size)}\n", style="bold")
    meaning = layer_meaning(p)
    if meaning:
        t.append(f"What it is: {meaning}\n", style="cyan")
    why = open_advice(p)
    if why:
        t.append(why.replace(" Showing it in its folder instead.", " o shows it in its folder.") + "\n",
                 style="yellow")
    t.append("\n")
    suf = p.suffix.lower()
    try:
        if suf in (".tif", ".tiff"):
            import numpy as np
            import rasterio
            with rasterio.open(p) as r:
                a = r.read(1, masked=True)
                t.append(f"GeoTIFF {r.width} × {r.height}, {r.count} band(s), {r.dtypes[0]}\n")
                t.append(f"CRS {r.crs}\nbounds {tuple(round(v, 5) for v in r.bounds)}\n")
                v = np.asarray(a.compressed(), dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    t.append(f"values: min {v.min():.3g}, median {np.median(v):.3g}, max {v.max():.3g} "
                             f"({v.size:,} valid cells)\n")
        elif suf in (".h5", ".he5"):
            import h5py
            with h5py.File(p, "r") as f:
                t.append("HDF5 datasets:\n")
                f.visititems(lambda name, obj: t.append(f"  {name}  {getattr(obj, 'shape', '')} {getattr(obj, 'dtype', '')}\n")
                             if hasattr(obj, "shape") else None)
                for k in ("FILE_TYPE", "REF_DATE", "START_DATE", "END_DATE", "UNIT"):
                    if k in f.attrs:
                        t.append(f"{k} = {f.attrs[k]}\n", style="dim")
        elif suf in (".png", ".jpg", ".jpeg", ".webp"):
            from PIL import Image
            with Image.open(p) as im:
                t.append(f"image {im.width} × {im.height}; o opens it\n")
        elif suf in TEXT or st.st_size < 2000 and suf == "":
            if st.st_size > 2_000_000:
                t.append("large text file: showing the first 200 lines\n\n", style="dim")
            with open(p, errors="replace") as fh:
                head = "".join([next(fh, "") for _ in range(200)])
            if suf in (".json", ".geojson") and st.st_size < 2_000_000:
                try:
                    head = json.dumps(json.loads(p.read_text()), indent=1)[:12000]
                except ValueError:
                    pass
            t.append(head)
        else:
            t.append("o opens it with the computer's own program.", style="dim")
    except Exception as exc:                                  # noqa: BLE001
        t.append(f"(no preview: {exc})", style="dim")
    return t
