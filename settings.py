"""Where things live on THIS machine. Nothing else in the pipeline hard-codes a
user's home, conda location or project folder.

  projects root   $INSAR_PROJECTS, else "projects_root" in the settings file,
                  else ~/insar_projects
  settings file   $INSAR_SETTINGS, else ~/.config/insar/settings.json
  conda           $CONDA_EXE (set by any activated conda env), else
                  `conda info --base`, else ~/miniforge3

Config files may use ~, $INSAR_PROJECTS and $INSAR_PIPELINE (this package's
folder) in any path, so one config works for everyone:

    project_dir: $INSAR_PROJECTS/site_a_pl
    deramp_exclude_geojson: $INSAR_PIPELINE/context/cmis_all_panels_union.geojson
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

PKG = Path(__file__).resolve().parent
log = logging.getLogger(__name__)


class SettingsError(ValueError):
    """The settings file exists but cannot be read as settings, so it is not
    overwritten (save)."""


def settings_file() -> Path:
    return Path(os.environ.get("INSAR_SETTINGS", Path.home() / ".config" / "insar" / "settings.json"))


def _read(f: Path) -> tuple:
    """(settings, problem). No file, or an empty one, is no settings and no
    problem; a file that cannot be read or is not a JSON object gives {} and
    what is wrong with it, in words."""
    try:
        text = f.read_text()
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as exc:                  # unreadable, or not text
        return {}, f"cannot be read ({exc})"
    if not text.strip():
        return {}, None
    try:
        doc = json.loads(text)
    except ValueError as exc:
        return {}, f"is not valid JSON ({exc})"
    if not isinstance(doc, dict):
        return {}, f"holds a JSON {type(doc).__name__}, not the {{\"name\": value}} object settings are"
    return doc, None


def problem() -> str | None:
    """What is wrong with the settings file, in words; None when it is fine or
    there is none (insar doctor reports it)."""
    f = settings_file()
    p = _read(f)[1]
    return f"{f} {p}" if p else None


_warned: set = set()


def load() -> dict:
    """The settings, {} when there are none. A file that does not parse also
    gives {} (callers must keep working) but no longer silently (16 Oct 2026,
    readiness review HND-03): the projects folder fell back to ~/insar_projects
    and the report identity went blank with nothing said. Warned once per
    file and problem, not on every one of the many calls."""
    f = settings_file()
    doc, why = _read(f)
    if why and (str(f), why) not in _warned:
        _warned.add((str(f), why))
        log.warning("Settings file %s %s. Using defaults until it is fixed: the projects folder is "
                    "$INSAR_PROJECTS or ~/insar_projects, and the report's firm and preparer are blank. Correct "
                    "the file by hand, or move it aside and set them again with insar doctor --projects DIR "
                    "--firm NAME --preparer NAME.", f, why)
    return doc


def save(**kw) -> Path:
    """Merge kw into the settings file. Refuses to overwrite a file that exists
    but does not parse (16 Oct 2026, readiness review HND-03): it used to be
    replaced by kw alone, losing every other setting in it."""
    f = settings_file()
    doc, why = _read(f)
    if why:
        raise SettingsError(f"Settings file {f} {why}. Not overwriting it, so the settings in it are not lost: "
                            "correct it by hand (or move it aside), then run this again.")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({**doc, **kw}, indent=1))
    return f


def projects_root() -> Path:
    p = os.environ.get("INSAR_PROJECTS") or load().get("projects_root") or (Path.home() / "insar_projects")
    return Path(os.path.expanduser(str(p)))


def configs_dir() -> Path:
    """New projects' configs: beside the projects, not inside the software."""
    return projects_root() / "configs"


@lru_cache(maxsize=1)
def conda_base() -> Path:
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if exe:
        return Path(exe).resolve().parents[1]
    try:
        out = subprocess.run(["conda", "info", "--base"], capture_output=True, text=True, timeout=30).stdout.strip()
        if out:
            return Path(out)
    except (OSError, subprocess.SubprocessError):
        pass
    return Path.home() / "miniforge3"


def conda_sh() -> Path:
    return conda_base() / "etc" / "profile.d" / "conda.sh"


def env_prefix(name: str) -> Path:
    return conda_base() if name in ("", "base") else conda_base() / "envs" / name


def identity() -> dict:
    """Who the reports say prepared them: {"firm", "preparer", "contact"} from
    the settings file ("" when unset; `insar-doctor --firm ... --preparer ...`
    writes them). Nothing in the package hard-codes a firm or a person."""
    s = load()
    return {k: str(s.get(k) or "").strip() for k in ("firm", "preparer", "contact")}


# ---------------------------------------------------------------- disk budget
GB = 1e9                                     # one unit everywhere: doctor, planner, dashboard


def _usage(path):
    return shutil.disk_usage(path)


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _existing(path) -> Path:
    """The path itself, or its nearest existing parent (a project folder that
    does not exist yet lives on its parent's filesystem)."""
    p = Path(os.path.expanduser(str(path))).absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _mount_kind(path: Path) -> str:
    """'root' (the WSL root filesystem, inside the vhdx), 'windows' (a drvfs
    drive such as /mnt/c: its free space is real) or 'other' (another disk)."""
    try:
        if os.stat(path).st_dev == os.stat("/").st_dev:
            return "root"
    except OSError:
        return "other"
    parts = Path(path).parts
    if len(parts) >= 3 and parts[1] == "mnt" and len(parts[2]) == 1:
        return "windows"
    return "other"


def _win_to_wsl(win: str) -> Path | None:
    """C:\\Users\\x\\... -> /mnt/c/Users/x/... without calling wslpath."""
    win = win.strip()
    if len(win) > 2 and win[1] == ":":
        return Path("/mnt") / win[0].lower() / win[3:].replace("\\", "/")
    return None


@lru_cache(maxsize=1)
def _vhdx_path() -> tuple:
    """(vhdx Path or None, how it was found). The registry entry of THIS distro
    first (exact even when the vhdx was moved off C: or several distros exist);
    then the folders `wsl --install` and the Store use, preferring the file
    written most recently (the running distro's disk is written all the time)."""
    distro = os.environ.get("WSL_DISTRO_NAME", "")
    reg = shutil.which("reg.exe")
    if reg and distro:
        try:
            out = subprocess.run([reg, "query", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Lxss", "/s"],
                                 capture_output=True, timeout=15).stdout.decode("utf-8", "replace")
            for block in out.split("HKEY_"):
                vals = {}
                for line in block.splitlines():
                    bits = line.split(None, 2)
                    if len(bits) == 3 and bits[1].startswith("REG_"):
                        vals[bits[0]] = bits[2].strip()
                if vals.get("DistributionName") == distro and vals.get("BasePath"):
                    base = vals["BasePath"].removeprefix("\\\\?\\")
                    p = _win_to_wsl(base.rstrip("\\") + "\\" + vals.get("VhdFileName", "ext4.vhdx"))
                    if p is not None and p.exists():
                        return p, f"registry entry for {distro}"
        except (OSError, subprocess.SubprocessError):
            pass
    import glob
    cands = [Path(c) for pat in ("/mnt/?/Users/*/AppData/Local/wsl/*/ext4.vhdx",
                                 "/mnt/?/Users/*/AppData/Local/Packages/*/LocalState/ext4.vhdx")
             for c in glob.glob(pat)]
    cands = [c for c in cands if c.is_file()]
    if not cands:
        return None, "no ext4.vhdx found under /mnt/<drive>/Users"
    best = max(cands, key=lambda c: c.stat().st_mtime)
    return best, ("found by search" if len(cands) == 1 else
                  f"most recently written of {len(cands)} WSL disks found (several distros)")


def _vhdx_sizes(vhdx) -> tuple:
    """(length, held) of the vhdx in bytes. `held` is what the file really
    takes on its Windows drive (st_blocks x 512, which drvfs reports from
    NTFS): less than the length when the file is sparse (WSL's sparseVhd,
    `fsutil sparse queryflag` -> "This file is set as sparse"), equal to it
    when it is not. Capped at the length (cluster rounding)."""
    st = Path(vhdx).stat()
    return st.st_size, min(st.st_blocks * 512, st.st_size)


def disk_budget_gb(path) -> dict:
    """
    How much a run under `path` can really write, in GB (1e9):
    {fs_free, vhdx_gb, used_gb, c_free, budget, source, host_drive,
     vhdx_held_gb, vhdx_headroom_gb, vhdx_sparse}.

    On WSL the root filesystem lives in an ext4.vhdx on a Windows drive.
    ext4 reports the vhdx's VIRTUAL free space (~1 TB by default), but only
    two things can take a new byte: blocks the vhdx file already holds on the
    drive that ext4 is not using, and the drive's own free space. So:

        budget = min(fs_free, free(drive) + max(0, held(vhdx) - used(/)))

    `held` is the file's allocation on the drive, not its length. For a
    non-sparse vhdx they are equal and this is the earlier formula. WSL now
    makes the vhdx SPARSE (sparseVhd): blocks ext4 frees are handed back to
    Windows, the length stays, and the gap between `used` and the length is
    no longer on the drive. MEASURED 7 Oct 2026: length 474.8 GB, held 377.1
    GB, used 367.6 GB, C: free 164.5 GB, so the real budget is 174.0 GB; the
    length-based (length - used) + C: free said 271.7 GB, ~98 GB too much
    (review_2026-10-08/disk/notes.md section 1). Under the old formula the
    figure fell 2 GB per GB written on a sparse disk, so the planner's gate
    passed runs that do not fit and the watchdog's 20 GB stop was luck.

    The figure can read high by the part of held - used that ext4 will not
    write into: its own metadata (journal, initialised inode tables), which
    `used` does not count - ext4 keeps 18.4 GB of its 1099.5 GB device as
    overhead (MEASURED 7 Oct: f_blocks 1081.1 GB) and how much of that is
    held is unknown - and slack in partly used vhdx blocks. Blocks ext4
    frees do not pile up there: "/" is mounted with `discard` (MEASURED 8
    Oct), so they go back to Windows at once. So the error is at most
    held - used (9.5 GB on 7 Oct) and C: free alone is a lower bound; the
    watchdog's 20 GB stop covers it while held - used stays below 20 GB.
    `held` (st_blocks x 512 through drvfs) equals Windows' own
    GetCompressedFileSizeW for the file to the byte (checked 8 Oct).

    Elsewhere - native Linux, a project on /mnt/<drive> or on another disk -
    budget = fs_free. When the vhdx cannot be found, budget = fs_free and
    `source` says so.
    """
    p = _existing(path)
    fs_free = _usage(p).free / GB
    out = {"fs_free": round(fs_free, 1), "vhdx_gb": None, "used_gb": None, "c_free": None,
           "budget": round(fs_free, 1), "source": "filesystem free space", "host_drive": None,
           "vhdx_held_gb": None, "vhdx_headroom_gb": None, "vhdx_sparse": None}
    if not _is_wsl():
        out["source"] = "filesystem free space (not WSL)"
        return out
    kind = _mount_kind(p)
    if kind == "windows":
        out["source"] = "Windows drive free space"
        return out
    if kind != "root":
        out["source"] = "filesystem free space (not on the WSL disk)"
        return out
    vhdx, how = _vhdx_path()
    if vhdx is None:
        out["source"] = f"filesystem only (vhdx not found: {how}) - the real budget may be smaller"
        return out
    try:
        length, held = _vhdx_sizes(vhdx)
        vh, held = length / GB, held / GB
        used = _usage("/").used / GB
        parts = Path(vhdx).parts                     # ('/', 'mnt', 'c', ...)
        host = Path("/", *parts[1:3])
        c_free = _usage(host).free / GB
    except OSError as exc:
        out["source"] = f"filesystem only (vhdx unreadable: {exc}) - the real budget may be smaller"
        return out
    headroom = max(0.0, held - used)
    budget = min(fs_free, headroom + c_free)
    drive = f"{parts[2].upper()}:"
    sparse = (vh - held) >= 0.1                      # a non-sparse file differs only by rounding
    out.update({"vhdx_gb": round(vh, 1), "used_gb": round(used, 1), "c_free": round(c_free, 1),
                "budget": round(budget, 1), "host_drive": drive,
                "vhdx_held_gb": round(held, 1), "vhdx_headroom_gb": round(headroom, 1), "vhdx_sparse": sparse,
                "source": f"WSL: {drive} free + space the {'sparse ' if sparse else ''}vhdx holds on {drive} "
                          f"that ext4 does not use ({how})"})
    return out


# ---------------------------------------------------------------- RAM and cores
_MEMINFO = Path("/proc/meminfo")


def machine_resources() -> dict:
    """{mem_gb, swap_gb, cores, mem_available_gb} of THIS Linux (on WSL: the VM,
    which by default gets half the PC's RAM - .wslconfig sets more). Memory in
    GiB, the unit .wslconfig's "memory=24GB" means, so a 24GB setting reads
    ~23.5 here (the kernel keeps the rest)."""
    mem = swap = avail = None
    try:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in _MEMINFO.read_text().splitlines() if l.strip()}
        mem, swap = mi["MemTotal"] / 1024 ** 2, mi.get("SwapTotal", 0) / 1024 ** 2
        avail = mi.get("MemAvailable", mi["MemTotal"]) / 1024 ** 2
    except (OSError, KeyError, ValueError, IndexError):
        try:
            mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
        except (ValueError, OSError, AttributeError):
            mem = 8.0                                   # unknown: assume a small machine
        swap, avail = 0.0, mem
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 4
    return {"mem_gb": round(mem, 2), "swap_gb": round(swap, 2), "cores": int(cores),
            "mem_available_gb": round(avail, 2)}


# Per-process peaks behind the pool sizes (all measured at Site A/Site B):
_OS_RESERVE_GB = 4.0        # WSL kernel, page cache for the disk-bound stages, the dashboard
_MERGE_PEAK_GB = 1.5        # merge_reference_secondary_slc: 1.27 GB peak RSS (median = max,
                            # site_b run_12 .done records), plus margin
_SNAPHU_PEAK_GB = 0.75      # full-frame snaphu 0.53 GB at 11x3 (0.93 at 8x2); cropped 0.11
_MINTPY_WORKER_GB = 1.4     # one dask worker per 1.4 GB: the 16 the P2 run used on 23 GB
_MIAPLPY_WORKER_GB = 0.5    # phase linking: ~120 MB per worker measured, x4 margin


def sized_parallelism(mem_gb: float, cores: int) -> dict:
    """
    Pool sizes this machine can hold, as PipelineConfig fields (every key is a
    config field, so a dict of them can go straight into a YAML). On the 23 GB
    / 24-core machine they were measured on this gives today's values exactly:
    stack_num_proc 12, stack_num_proc_topo 3, stack_num_proc_unwrap 24,
    mintpy_num_workers 16, mintpy_max_memory_gb 16, miaplpy_num_processors 20.
    On a 16 GB laptop (WSL default: ~7.8 GB) it gives 2 / 1 / 5 / 5 / 5 / 4,
    which fit where today's values were OOM-killed. Speed only, never numbers.
    """
    mem = max(float(mem_gb or 0), 1.0)
    cores = max(int(cores or 1), 1)
    room = max(mem - _OS_RESERVE_GB, 0.0)
    clamp = lambda v, lo, hi: int(max(lo, min(hi, v)))          # noqa: E731
    return {
        "stack_num_proc": clamp(room // _MERGE_PEAK_GB, 1, 12),
        # topo/geo2rdr/resample run ~8 threads each (measured): one per 8 cores
        "stack_num_proc_topo": clamp(cores // 8, 1, 3),
        "stack_num_proc_unwrap": clamp(min(cores, room // _SNAPHU_PEAK_GB), 1, 24),
        "mintpy_num_workers": clamp(min(cores, mem // _MINTPY_WORKER_GB), 1, 16),
        "mintpy_max_memory_gb": float(clamp(int(0.7 * mem), 2, 16)),
        # leave a few cores free for the rest of the machine
        "miaplpy_num_processors": clamp(min(max(min(4, cores), cores - 4), room // _MIAPLPY_WORKER_GB), 1, 20),
    }


def export_env() -> None:
    """Make $INSAR_PIPELINE and $INSAR_PROJECTS available to config paths and
    child processes (never overriding what the user set)."""
    os.environ.setdefault("INSAR_PIPELINE", str(PKG))
    os.environ.setdefault("INSAR_PROJECTS", str(projects_root()))


export_env()
