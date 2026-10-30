"""Is this installation ready to run? `insar doctor`

Checks what a run needs, in the order it needs it, and says how to fix each
problem. Exit status 0 when nothing is missing.

    insar doctor                       everything, including the test suite (~1 min)
    insar doctor --quick               skip the test suite
    insar doctor --projects /path      move the projects folder
    insar doctor --firm "Name" --preparer "Name"   who the reports say prepared them
"""
from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from . import settings

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"

# A phase-linked decade record peaks near 300 GB (measured: Site A and
# Site B, ~270 dates); 350 leaves room for the products beside it.
PHASE_LINKED_PEAK_GB = 300
DISK_WARN_GB = 350
# RAM the WSL VM needs (GiB, as .wslconfig counts it). MEASURED: WSL's default
# of half the PC's RAM gave 15.9 GB on Kyle's 32 GB PC, and a run was OOM-killed
# 16 s after assembling; 24GB in .wslconfig has run every stage since. Below 12
# the sized pools are too small to finish a decade stack in a working week.
MEM_FAIL_GB, MEM_WARN_GB = 12.0, 20.0


def _line(mark, msg, fix=""):
    print(f"  {mark} {msg}" + (f"\n      → {fix}" if fix else ""))


# ---------------------------------------------------------------- disk
def disk_checks(root) -> list:
    """[(ok, msg, fix, warn)] for the projects folder, measured against the REAL
    budget (settings.disk_budget_gb: on WSL the Windows drive's free space plus
    what the vhdx already holds on it that ext4 does not use), not ext4's
    virtual free space, which overstated it by ~320 GB on 5 Oct 2026 (743 GB
    shown, 422 GB real), nor the vhdx file's length, which on a sparse vhdx
    overstated it by ~98 GB on 7 Oct 2026 (272 GB shown, 174 GB real)."""
    b = settings.disk_budget_gb(root)
    drive = b.get("host_drive") or "C:"
    if b.get("vhdx_held_gb") is not None:
        sparse = (f"; sparse file, {b['vhdx_gb']:.0f} GB long, so its length is not space on {drive}"
                  if b.get("vhdx_sparse") else "")
        where = (f"{b['budget']:.0f} GB real budget: {drive} {b['c_free']:.0f} GB free + "
                 f"{b['vhdx_headroom_gb']:.0f} GB unused inside the WSL disk (it holds {b['vhdx_held_gb']:.0f} GB "
                 f"on {drive}, ext4 uses {b['used_gb']:.0f} GB{sparse}); Linux shows {b['fs_free']:.0f} GB")
    elif b.get("vhdx_gb") is not None:
        where = (f"{b['budget']:.0f} GB real budget: {drive} {b['c_free']:.0f} GB free, "
                 f"WSL disk {b['vhdx_gb']:.0f} GB holding {b['used_gb']:.0f} GB; Linux shows {b['fs_free']:.0f} GB")
    else:
        where = f"{b['budget']:.0f} GB free ({b['source']})"
    out = [(os.access(root, os.W_OK), f"projects folder {root} ({where})",
            "set another with: insar doctor --projects /path  (or INSAR_PROJECTS)", False)]
    if "vhdx not found" in b["source"] or "vhdx unreadable" in b["source"]:
        out.append((False, "WSL disk file (ext4.vhdx) not found, so free space may be overstated",
                    "check free space on the Windows drive that holds WSL yourself (df -h /mnt/c)", True))
    out.append((b["budget"] > DISK_WARN_GB, f"room for a phase-linked run (~{PHASE_LINKED_PEAK_GB} GB peak; "
                f"{b['budget']:.0f} GB available)",
                "fine for DISP-S1 screening; free space on the Windows drive or move the projects folder "
                "for phase-linked runs", True))
    return out


# ---------------------------------------------------------------- memory
def _windows_home() -> Path | None:
    """C:\\Users\\<you> as seen from WSL: from the WSL disk's path, else the only user folder with a .wslconfig."""
    try:
        vhdx, _how = settings._vhdx_path()
    except Exception:                                          # noqa: BLE001
        vhdx = None
    if vhdx is not None and "Users" in Path(vhdx).parts:
        parts = Path(vhdx).parts
        return Path(*parts[:parts.index("Users") + 2])
    homes = [p.parent for p in Path("/mnt/c/Users").glob("*/.wslconfig")] if Path("/mnt/c/Users").exists() else []
    return homes[0] if len(homes) == 1 else None


def memory_checks(res: dict | None = None, wsl: bool | None = None) -> list:
    """[(ok, msg, fix, warn)] for RAM: fail below 12 GB, warn below 20 GB, and
    say exactly how to give WSL more."""
    res = res or settings.machine_resources()
    wsl = ("microsoft" in platform.release().lower()) if wsl is None else wsl
    mem, cores, swap = res["mem_gb"], res["cores"], res.get("swap_gb", 0.0)
    pools = settings.sized_parallelism(mem, cores)
    msg = (f"memory {mem:.1f} GB, swap {swap:.0f} GB, {cores} cores - pools sized for it: "
           f"{pools['stack_num_proc']} merges, {pools['stack_num_proc_unwrap']} unwraps, "
           f"MintPy {pools['mintpy_num_workers']} workers / {pools['mintpy_max_memory_gb']:.0f} GB")
    if wsl:
        home = _windows_home()
        where = f"{home}/.wslconfig" if home else "C:\\Users\\<you>\\.wslconfig"
        cur = ""
        try:
            lines = (home / ".wslconfig").read_text().splitlines() if home else []
            cur = next((ln.strip() for ln in lines if ln.strip().lower().startswith("memory")), "")
        except OSError:
            pass
        fix = (f"WSL gets half the PC's RAM by default. In {where} put\n"
               "          [wsl2]\n"
               "          memory=24GB      (your PC's RAM minus 6-8 GB; 24GB on a 32 GB PC)\n"
               "          swap=16GB\n"
               "        then in Windows PowerShell run: wsl --shutdown   and reopen Ubuntu"
               + (f"\n        (now: {cur})" if cur else ""))
    else:
        fix = "this machine needs more RAM; at least 20 GB is recommended for phase-linked runs"
    if mem < MEM_FAIL_GB:
        return [(False, msg + f" - below {MEM_FAIL_GB:.0f} GB", fix, False)]
    if mem < MEM_WARN_GB:
        return [(False, msg + f" - below the {MEM_WARN_GB:.0f} GB the pipeline was measured on", fix, True)]
    return [(True, msg, "", False)]


def pool_checks(cfg_dir: Path, res: dict | None = None) -> list:
    """Configs that set pool sizes this machine cannot hold (a project copied
    from a bigger PC): one warning each. Values below the machine's are fine."""
    import yaml
    res = res or settings.machine_resources()
    sized = settings.sized_parallelism(res["mem_gb"], res["cores"])
    out = []
    for f in sorted(Path(cfg_dir).glob("*.yaml")) if Path(cfg_dir).exists() else []:
        try:
            raw = yaml.safe_load(f.read_text()) or {}
        except (OSError, yaml.YAMLError):
            continue
        over = [f"{k} {raw[k]} (this machine: {v:g})" for k, v in sized.items()
                if isinstance(raw.get(k), (int, float)) and float(raw[k]) > float(v)]
        if over:
            out.append((False, f"{f.name} sets more than this machine holds: " + ", ".join(over),
                        f"delete those lines from {f} (unset = sized for the machine) or lower them", True))
    return out


# The fix for a project whose own mini-stack / overlap fails the patch check.
# The boundary error grows with the overlap as well as failing without one
# (re-measured 7 Oct 2026, mini-stack 10: lag 0 ~22x; 1, 3 and 6 pass; 7 and 9 fail),
# so 'raise it' was wrong advice for a lag of 7 or more.
MIAPLPY_TIME_LAG_FIX = ("set miaplpy_time_lag to 3, the tested default, in those configs (the check fails at 0 "
                        "and from about 7 with mini-stack 10)")


def miaplpy_settings_in(cfg_dir: Path) -> dict:
    """{(miaplpy_ministack_size, miaplpy_time_lag): [config names]} for the
    phase-linked configs in cfg_dir whose pair differs from the defaults the
    bare patch check tests. Read from the YAML only (a config that does not
    load is the config check's business, not this one's)."""
    import dataclasses
    import yaml
    from .config import PipelineConfig
    dflt = {f.name: f.default for f in dataclasses.fields(PipelineConfig)
            if f.name in ("miaplpy_ministack_size", "miaplpy_time_lag")}
    base = (int(dflt.get("miaplpy_ministack_size", 10)), int(dflt.get("miaplpy_time_lag", 3)))
    out: dict = {}
    for f in sorted(Path(cfg_dir).glob("*.yaml")) if Path(cfg_dir).exists() else []:
        try:
            raw = yaml.safe_load(f.read_text()) or {}
        except (OSError, yaml.YAMLError):
            continue
        if str(raw.get("backend", "")).lower() != "miaplpy":
            continue
        try:
            pair = (int(raw.get("miaplpy_ministack_size", base[0])), int(raw.get("miaplpy_time_lag", base[1])))
        except (TypeError, ValueError):
            continue
        if pair != base:
            out.setdefault(pair, []).append(f.name)
    return out


# ---------------------------------------------------------------- known answers
def snaphu_known_answer(exe: str) -> tuple:
    """(ok, detail): unwrap a synthetic ramp-plus-bowl whose answer is known and
    compare. Catches a snaphu that runs but unwraps wrongly (the GCC 14 build
    trap), which a version string does not. Measured: residual 2e-5 rad."""
    import numpy as np
    h, w = 64, 80
    y, x = np.mgrid[0:h, 0:w]
    truth = 0.45 * x + 0.3 * y + 4.0 * np.exp(-((x - 40) ** 2 + (y - 32) ** 2) / 200.0)   # 46 rad, 7 cycles
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        np.exp(1j * truth).astype(np.complex64).tofile(d / "in.c8")
        (d / "s.conf").write_text(f"OUTFILE {d / 'out.unw'}\nINFILEFORMAT COMPLEX_DATA\nOUTFILEFORMAT FLOAT_DATA\n"
                                  "STATCOSTMODE SMOOTH\nINITMETHOD MST\n")
        try:
            r = subprocess.run([exe, "-f", str(d / "s.conf"), str(d / "in.c8"), str(w)],
                               capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"did not run ({exc})"
        out = d / "out.unw"
        if r.returncode or not out.exists() or out.stat().st_size != h * w * 4:
            return False, f"failed (exit {r.returncode}): {(r.stderr or r.stdout)[-200:].strip()}"
        u = np.fromfile(out, np.float32).reshape(h, w).astype(float)
    dif = u - truth
    err = float(np.nanmax(np.abs(dif - np.nanmedian(dif))))
    return err < 0.1, f"known-answer unwrap residual {err:.1e} rad (limit 0.1)"


def suite_check(tests_dir: Path | None = None, timeout: int = 1800) -> tuple:
    """(ok, summary): run the package's own test suite in this interpreter.
    The golden tests (need the delivered project on disk) are left out.
    The suite takes about 9 minutes on a fast PC (16 Oct 2026: ~1,960 tests),
    so the limit is 30 minutes; at 10 a slower PC's install could end "not
    ready" with nothing wrong."""
    tests_dir = Path(tests_dir or settings.PKG / "tests")
    if not tests_dir.exists():
        return False, f"no tests at {tests_dir}"
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", str(tests_dir), "-q", "-x", "-p", "no:cacheprovider",
                            "-m", "not golden"], capture_output=True, text=True, timeout=timeout,
                           cwd=str(tests_dir.parent.parent))
    except subprocess.TimeoutExpired:
        return False, f"did not finish in {timeout} s"
    except OSError as exc:
        return False, f"could not run pytest ({exc})"
    tail = [ln for ln in (r.stdout + r.stderr).strip().splitlines() if ln.strip()]
    summary = tail[-1].strip("= ") if tail else f"exit {r.returncode}"
    first = next((ln.split()[1] for ln in tail if ln.startswith(("FAILED ", "ERROR ")) and len(ln.split()) > 1), "")
    if first:
        summary += f" (first: {first.split('/')[-1]})"
    # exit 5 = no tests collected (every test deselected) - not a pass
    return r.returncode == 0, summary


# ---------------------------------------------------------------- Earthdata
EARTHDATA_UNREACHABLE = "the Earthdata / ASF servers did not answer; try again later"


def earthdata_login_status() -> tuple[str, str]:
    """(code, message) for the stored NASA Earthdata login: data_acquisition's
    ok / missing / rejected / not_authorised / unreachable. 'unreachable' (a
    server error from Earthdata or ASF, or no answer at all) says nothing
    about the login, so it must not send anyone to change ~/.netrc.
    check_earthdata_login wraps the coded status as ok-or-not; when it has
    been replaced by a stand-in, the stand-in's answer decides and a
    failure counts as a login problem."""
    from . import data_acquisition as da
    check = da.check_earthdata_login
    status = getattr(da, "_earthdata_login_status", None)
    if status is None or getattr(check, "__module__", None) != da.__name__:
        good, msg = check()
        return ("ok" if good else "rejected"), msg
    return status()


def earthdata_checks(netrc: Path | None = None) -> list:
    """[(ok, msg, fix, warn)]: a NASA Earthdata login in ~/.netrc, then the
    login tried the way the downloads use it. Wrong or missing credentials
    and ASF access not authorised fail; an outage is a warning, with no
    ~/.netrc fix."""
    netrc = netrc or Path.home() / ".netrc"
    has_netrc = netrc.exists() and "urs.earthdata.nasa.gov" in netrc.read_text(errors="replace")
    out = [(has_netrc, "NASA Earthdata login in ~/.netrc",
            "free account at urs.earthdata.nasa.gov; install/install.sh writes the file", False)]
    if not has_netrc:
        return out
    try:
        code, message = earthdata_login_status()
    except (ImportError, AttributeError):
        return out                       # no data_acquisition here to try the login with
    except Exception as exc:                                   # noqa: BLE001 - offline: cannot tell
        return out + [(False, f"Earthdata login not verified ({exc})", "check the network and rerun", True)]
    if code == "unreachable":
        out.append((False, f"Earthdata login not checked: {EARTHDATA_UNREACHABLE} ({str(message)[:200]})", "",
                    True))
    elif code == "not_authorised":
        out.append((False, f"Earthdata login fails: {message}",
                    "sign in once at https://search.asf.alaska.edu and accept the terms, then rerun insar doctor",
                    False))
    else:
        out.append((code == "ok", f"Earthdata login works ({message})" if code == "ok"
                    else f"Earthdata login fails: {message}",
                    "fix the user name / password in ~/.netrc (urs.earthdata.nasa.gov)", False))
    return out


def _cds_problems():
    """check_weather_credentials as [(level, text)]: newer versions return
    {"level": "fail"|"warn", "text": ...}, older ones plain strings (= fail)."""
    from types import SimpleNamespace
    try:
        from .stack_backend import check_weather_credentials
        probs = check_weather_credentials(SimpleNamespace(mintpy_tropo_method="pyaps"))
    except Exception as exc:                                   # noqa: BLE001
        return [("warn", f"could not check ({exc})")]
    out = []
    for p in probs or ():
        if isinstance(p, dict):
            out.append((str(p.get("level", "fail")), str(p.get("text", p))))
        else:
            out.append(("fail", str(p)))
    return out


def main() -> int:
    argv = sys.argv
    try:
        if "--projects" in argv:
            settings.save(projects_root=str(Path(argv[argv.index("--projects") + 1]).expanduser().resolve()))
            os.environ.pop("INSAR_PROJECTS", None)
        for flag in ("--firm", "--preparer", "--contact"):
            if flag in argv:
                settings.save(**{flag[2:]: argv[argv.index(flag) + 1]})
    except settings.SettingsError as exc:            # a settings file that does not parse is never overwritten
        print(f"\n  {exc}\n")
        return 1
    problems = 0

    def check(ok, msg, fix="", warn=False):
        nonlocal problems
        if not ok and not warn:
            problems += 1
        _line(OK if ok else (WARN if warn else BAD), msg, "" if ok else fix)

    try:
        from . import version
        v = version.describe()
        ver = f"version {v.get('version')} (code {v.get('code_hash')})"
    except Exception:                                          # noqa: BLE001 - version.py is optional
        ver = "version unknown"
    print(f"\nInSAR pipeline - installation check - {ver}\n")
    wsl = "microsoft" in platform.release().lower()
    check(sys.platform.startswith("linux"), f"Linux ({'WSL2 on Windows' if wsl else platform.platform()})",
          "ISCE2 and MintPy run on Linux; on Windows use WSL2 with Ubuntu")
    base = settings.conda_base()
    check((base / "bin" / "conda").exists() or (base / "condabin" / "conda").exists(), f"conda at {base}",
          "run install/install.sh (it installs Miniforge if needed)")
    # burst2safe: every download runs `conda run -n burst2safe` (stack_use_burst2safe)
    for env in ("insar", "miaplpy", "burst2safe"):
        check(settings.env_prefix(env).exists(), f"conda environment '{env}'", "run install/install.sh")

    # this interpreter: the insar env (imported by name: the import IS the check)
    try:
        importlib.import_module("isce")
        mintpy = importlib.import_module("mintpy")
        textual = importlib.import_module("textual")
        check(True, f"ISCE2, MintPy {mintpy.__version__}, Textual {textual.__version__} import")
    except Exception as exc:                                   # noqa: BLE001
        check(False, f"core packages import ({exc})", "activate the insar env, or rerun install/install.sh")
    try:
        from .stack_backend import ISCE_STACK
        check((ISCE_STACK / "stackSentinel.py").exists(), f"ISCE2 topsStack scripts ({ISCE_STACK})",
              "reinstall isce2 in the insar env, or set ISCE_STACK to its share/isce2 folder")
    except Exception as exc:                                   # noqa: BLE001
        check(False, f"ISCE2 topsStack scripts ({exc})", "rerun install/install.sh")

    try:
        from .unwrap_crop import find_snaphu
        sn = find_snaphu("")
        r = subprocess.run([sn], capture_output=True, text=True, timeout=30)
        check("2.0.7" in r.stdout + r.stderr, f"snaphu 2.0.7 ({sn})", "run install/build_snaphu.sh")
        ok, detail = snaphu_known_answer(sn)
        check(ok, f"snaphu unwraps correctly ({detail})", "rebuild it: install/build_snaphu.sh")
    except Exception as exc:                                   # noqa: BLE001
        check(False, f"snaphu 2.0.7 ({exc})", "run install/build_snaphu.sh")

    # the results map's page and every file it loads (results_viewer.ASSETS): a
    # release that left the page out broke 'm' on an install (BLD-04)
    try:
        from . import results_viewer as rv
        missing = ([] if (rv.PKG / rv.PAGE).is_file() else [rv.PAGE]) + [k for k, f in rv.ASSETS.items() if not f.is_file()]
        check(not missing, "results map page and files" + (f" (missing: {', '.join(missing[:4])})" if missing else ""),
              "reinstall the pipeline release (install/make_release.sh ships them)")
    except Exception as exc:                                   # noqa: BLE001
        check(False, f"results map files ({exc})", "reinstall the pipeline release")
    import shutil
    node = shutil.which("node")
    check(node is not None, f"Node.js, for the tests of the results map's browser code ({node or 'not found'})",
          "rerun install/install.sh: the insar env pins nodejs", warn=True)

    try:
        from .miaplpy_backend import verify_patched_miaplpy
        info = verify_patched_miaplpy("miaplpy")
        check(True, f"patched MiaplPy {info.get('miaplpy_version')} (boundary error {info.get('ratio')}x inside)")
        # the projects' own mini-stack and window overlap, when they differ from
        # the defaults the bare check uses: overlap 0 brings the datum defect
        # back (boundary error ~22x) where the default check passes
        for (ms, tl), names in sorted(miaplpy_settings_in(settings.configs_dir()).items()):
            try:
                i2 = verify_patched_miaplpy("miaplpy", ministack=ms, time_lag=tl)
                check(True, f"patched MiaplPy with mini-stack {ms} / overlap {tl} ({', '.join(names[:3])}): "
                            f"boundary error {i2.get('ratio')}x inside")
            except Exception as exc:                           # noqa: BLE001
                check(False, f"MiaplPy with mini-stack {ms} / overlap {tl} ({', '.join(names[:3])}): "
                             f"{str(exc)[:120]}", MIAPLPY_TIME_LAG_FIX)
    except Exception as exc:                                   # noqa: BLE001
        check(False, f"patched MiaplPy ({str(exc)[:120]})", "run patches/miaplpy/build_miaplpy.sh miaplpy")

    for ok, msg, fix, warn in earthdata_checks():
        check(ok, msg, fix, warn=warn)
    check((Path.home() / ".cdsapirc").exists(), "Copernicus CDS key in ~/.cdsapirc (ERA5 weather)",
          "free account at cds.climate.copernicus.eu; install/install.sh writes the file", warn=True)
    if (Path.home() / ".cdsapirc").exists():
        probs = _cds_problems()
        fails = [t for lvl, t in probs if lvl == "fail"]
        warns = [t for lvl, t in probs if lvl != "fail"]
        if fails:
            check(False, "ERA5 access: " + "; ".join(fails),
                  "sign in at https://cds.climate.copernicus.eu and accept the licence named above "
                  "(then rerun insar doctor)")
        elif warns:
            check(False, "ERA5 access not fully verified: " + "; ".join(warns), "", warn=True)
        else:
            check(True, "ERA5 access (CDS key and licence)")

    # machine: RAM and cores (WSL gets half the PC's RAM unless .wslconfig says otherwise)
    res = settings.machine_resources()
    for ok, msg, fix, warn in memory_checks(res, wsl):
        check(ok, msg, fix, warn)

    root = settings.projects_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        for ok, msg, fix, warn in disk_checks(root):
            check(ok, msg, fix, warn)
    except OSError as exc:
        check(False, f"projects folder {root} ({exc})", "set INSAR_PROJECTS to a writable folder")
    for ok, msg, fix, warn in pool_checks(settings.configs_dir(), res):
        check(ok, msg, fix, warn)

    # a settings file that does not parse reads as no settings (16 Oct 2026,
    # readiness review HND-03): say so, or the defaults below look chosen
    bad = settings.problem()
    if bad:
        check(False, f"settings file {bad}", "correct it by hand, or move it aside and set the projects folder and "
              "report identity again; until then both are the defaults")
    who = settings.identity()
    check(bool(who["firm"] and who["preparer"]),
          f"report identity: {who['firm'] or '(firm not set)'}, prepared by {who['preparer'] or '(not set)'}",
          'insar doctor --firm "Your Firm" --preparer "Your Name"', warn=True)

    if "--quick" not in argv:
        print("  … running the test suite")
        ok, summary = suite_check()
        check(ok, f"test suite: {summary}",
              "the install is not computing what it should: rerun install/install.sh, then insar doctor")

    print(f"\n  {'Ready.' if not problems else f'{problems} problem(s) to fix.'}  Settings: {settings.settings_file()}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
