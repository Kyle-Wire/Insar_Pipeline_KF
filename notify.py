"""Tell a person when an unattended run ends.

Runs take a day; nobody watches the dashboard overnight. Site B crashed at
06:35 on 29 Sep and was found at 08:30. send() raises a Windows desktop
notification (a "toast") from WSL through powershell.exe, which WSL can run
directly (/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe), and
appends the message to ~/.insar_pipeline_notifications.log so it is never
lost. It never raises: a notification that cannot be shown must not turn a
finished run into a failed one.

Settings file (settings.py) key "notify":
    {"toast": true}        default; false turns the desktop notification off
    {"command": "..."}     optional extra shell command; {title} and {text}
                           are substituted, shell-quoted (e.g. a mail or chat hook)
Environment INSAR_NOTIFY=off disables everything but the log line (tests set it).
"""
from __future__ import annotations

import base64
import logging
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from xml.sax.saxutils import escape

log = logging.getLogger(__name__)

# INSAR_NOTIFY_LOG moves it: the test suite points it (and INSAR_NOTIFY=off)
# at a temporary file, so no test - not even one that runs the pipeline in a
# subprocess - writes into the real log or raises a toast (16 Oct 2026).
LOGFILE = Path(os.environ.get("INSAR_NOTIFY_LOG") or Path.home() / ".insar_pipeline_notifications.log")
_PS_CANDIDATES = ("/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe",
                  "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
# Windows only shows toasts from a registered app id; PowerShell's own is
# registered on every Windows 10/11 install, so no extra module (BurntToast)
# is needed.
_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


def powershell() -> str | None:
    exe = shutil.which("powershell.exe")
    if exe:
        return exe
    return next((p for p in _PS_CANDIDATES if os.path.exists(p)), None)


def _settings() -> dict:
    try:
        from . import settings
        v = settings.load().get("notify", {})
        return v if isinstance(v, dict) else {"toast": bool(v)}
    except Exception:                                        # noqa: BLE001
        return {}


def toast_script(title: str, text: str) -> str:
    """The PowerShell that shows one toast. Title and text go in XML-escaped
    and inside single-quoted PowerShell strings (quotes doubled), so nothing in
    an error message can run as code."""
    def ps(s: str) -> str:
        return "'" + s.replace("'", "''") + "'"
    xml = ("<toast><visual><binding template='ToastGeneric'>"
           f"<text>{escape(title[:120])}</text><text>{escape(text[:600])}</text>"
           "</binding></visual></toast>")
    return (
        "$ErrorActionPreference='Stop';"
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null;"
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] | Out-Null;"
        "$x = New-Object Windows.Data.Xml.Dom.XmlDocument;"
        f"$x.LoadXml({ps(xml)});"
        "$t = New-Object Windows.UI.Notifications.ToastNotification $x;"
        f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier({ps(_APP_ID)}).Show($t);"
        "Write-Output 'shown'")


def _toast(title: str, text: str, timeout: float = 30.0) -> bool:
    exe = powershell()
    if not exe:
        return False
    enc = base64.b64encode(toast_script(title, text).encode("utf-16-le")).decode()
    try:
        p = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                            "-EncodedCommand", enc], capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL, cwd="/")
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("Desktop notification not shown: %s", exc)
        return False
    ok = p.returncode == 0 and "shown" in (p.stdout or "")
    if not ok:
        log.info("Desktop notification not shown (rc %s): %s", p.returncode, (p.stderr or "").strip()[:200])
    return ok


def send(title: str, text: str) -> dict:
    """Log line + desktop toast (+ optional command). Returns what happened,
    {"logged", "toast", "command"}; never raises."""
    out = {"logged": False, "toast": False, "command": None}
    try:
        with open(LOGFILE, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {title}  |  {' '.join(text.split())}\n")
        out["logged"] = True
    except OSError:
        pass
    try:
        if os.environ.get("INSAR_NOTIFY", "").lower() in ("0", "off", "false", "no"):
            return out
        s = _settings()
        if s.get("toast", True):
            out["toast"] = _toast(title, text)
        cmd = s.get("command")
        if cmd:
            line = cmd.replace("{title}", shlex.quote(title)).replace("{text}", shlex.quote(text))
            out["command"] = subprocess.run(line, shell=True, timeout=60, capture_output=True).returncode
    except Exception as exc:                                 # noqa: BLE001
        log.info("Notification failed: %s", exc)
    return out
