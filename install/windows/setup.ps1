<#
Set up the InSAR pipeline on this Windows PC, start to finish:

  1. Ubuntu in WSL2 (asks for administrator rights once; Windows restarts)
  2. WSL's memory (.wslconfig), sized for this PC
  3. the pipeline inside Ubuntu (install/install.sh: about 30 minutes,
     it asks for the Earthdata login and the Copernicus CDS token)
  4. an "InSAR Pipeline" icon on the desktop and in the Start menu

Double-click "Setup InSAR Pipeline.cmd" beside this file, or run:

  powershell -ExecutionPolicy Bypass -File setup.ps1 [-Release <file.tar.gz>] [-Distro Ubuntu] [-IconOnly]

Safe to run again: every step that is already done is skipped or asked about.
-Release: the release file to install (default: the newest
insar_pipeline_*.tar.gz beside this script). -IconOnly: only step 4, for a PC
where the pipeline is already installed.
#>
param([string]$Release = "", [string]$Distro = "Ubuntu", [switch]$IconOnly)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Say($m) { Write-Host ""; Write-Host "== $m" -ForegroundColor Cyan }
function Ok($m) { Write-Host "   OK  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "   !!  $m" -ForegroundColor Yellow }
function Ask($q) { return ((Read-Host "   $q [y/N]") -match '^(y|yes)$') }
function Done($code) { Write-Host ""; Read-Host "Press Enter to close" | Out-Null; exit $code }

# wsl.exe writes UTF-16: strip the NULs PowerShell 5 leaves in the text
function Distros {
    $out = & wsl.exe -l -q 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return @() }
    return @($out | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ })
}
function InUbuntu([string]$cmd) {
    $out = & wsl.exe -d $Distro -- bash -lc $cmd 2>$null
    return (($out | ForEach-Object { ($_ -replace "`0", "") }) -join "`n").Trim()
}
function IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

Write-Host "InSAR pipeline setup" -ForegroundColor White

if (-not $IconOnly) {
    # ------------------------------------------------------------ 1. Ubuntu in WSL2
    Say "1. Ubuntu (WSL2)"
    $hasWsl = [bool](Get-Command wsl.exe -ErrorAction SilentlyContinue)
    if (-not $hasWsl -or -not ((Distros) -contains $Distro)) {
        Warn "$Distro is not installed in WSL yet."
        Write-Host "   Windows will ask for administrator rights, install it, and may ask to restart."
        Write-Host "   Afterwards: open '$Distro' from the Start menu once, choose a username and"
        Write-Host "   password (only for Linux), close it, and run this setup again."
        if (-not (Ask "Install $Distro now?")) { Done 1 }
        $elev = "-NoProfile -ExecutionPolicy Bypass -Command `"wsl.exe --install -d $Distro; Read-Host 'Done. Restart if Windows asked, then open $Distro once. Press Enter'`""
        if (IsAdmin) { Invoke-Expression "wsl.exe --install -d $Distro" }
        else { Start-Process powershell.exe -Verb RunAs -ArgumentList $elev -Wait }
        Done 0
    }
    $user = InUbuntu "whoami"
    if (-not $user) {
        Warn "$Distro does not answer. Open '$Distro' from the Start menu once (it finishes setting itself up), then run this again."
        Done 1
    }
    if ($user -eq "root") {
        Warn "$Distro runs as root: open '$Distro' from the Start menu once and create your user, then run this again."
        Done 1
    }
    Ok "$Distro is ready (Linux user $user)"

    # ------------------------------------------------------------ 2. WSL memory
    Say "2. Memory for WSL"
    $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $want = [math]::Max(12, $ram - 8)
    $wc = Join-Path $env:USERPROFILE ".wslconfig"
    $text = if (Test-Path $wc) { Get-Content $wc -Raw } else { "" }
    if ($text -match '(?m)^\s*memory\s*=\s*(\S+)') {
        Ok "kept your .wslconfig (memory=$($Matches[1])); insar doctor checks it"
    } else {
        if ($ram -lt 16) { Warn "this PC has $ram GB; the pipeline needs 12 GB for WSL and 32 GB in the PC is recommended" }
        $lines = "memory=${want}GB`r`nswap=16GB`r`n"
        if (Test-Path $wc) { Copy-Item $wc "$wc.bak-$(Get-Date -Format yyyyMMdd-HHmm)" }
        if ($text -match '(?m)^\s*\[wsl2\]\s*$') { $text = $text -replace '(?m)^(\s*\[wsl2\]\s*)$', "`$1`r`n$lines" }
        else { $text = $text + "`r`n[wsl2]`r`n" + $lines }
        Set-Content -Path $wc -Value $text.Trim() -Encoding ASCII
        Ok "wrote memory=${want}GB, swap=16GB to $wc (this PC has $ram GB)"
        Write-Host "   WSL must restart to use it. That stops everything running in Ubuntu,"
        Write-Host "   a pipeline run included (it can be resumed, but it loses its place)."
        if (Ask "Nothing is running in Ubuntu now: restart WSL?") {
            & wsl.exe --shutdown
            Start-Sleep -Seconds 3
            Ok "WSL restarted"
        } else { Warn "restart it later with: wsl --shutdown (when no run is going)" }
    }

    # ------------------------------------------------------------ 3. the pipeline
    Say "3. The pipeline in $Distro"
    $installed = (InUbuntu "test -x ~/.local/bin/insar && test -f ~/insar_pipeline/install/install.sh && echo yes") -eq "yes"
    $run = $true
    if ($installed) {
        Ok "already installed (~/insar_pipeline)"
        $run = Ask "Run its installer again (finished steps are skipped; use it after an update)?"
    } elseif ((InUbuntu "test -f ~/insar_pipeline/install/install.sh && echo yes") -ne "yes") {
        if (-not $Release) {
            $cand = Get-ChildItem -Path $Here -Filter "insar_pipeline_*.tar.gz" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($cand) { $Release = $cand.FullName }
        }
        if (-not $Release -or -not (Test-Path $Release)) {
            Warn "no release file found. Put insar_pipeline_<version>.tar.gz (and its .sha256) beside this"
            Write-Host "   script and run it again, or clone the repository inside Ubuntu (install/README.md,"
            Write-Host "   'From the GitHub repository') and run this again for the icon."
            Done 1
        }
        # copied into the Linux home first: the commands below then name a file
        # without spaces, and no quotes have to survive the trip through wsl.exe
        $name = Split-Path -Leaf $Release
        if ($name -notmatch '^insar_pipeline_[A-Za-z0-9._-]+\.tar\.gz$') { Warn "unexpected release file name: $name"; Done 1 }
        $home_ = InUbuntu "wslpath -w ~"
        Write-Host "   copying $name into $home_"
        Copy-Item $Release (Join-Path $home_ $name) -Force
        if (Test-Path "$Release.sha256") {
            Copy-Item "$Release.sha256" (Join-Path $home_ "$name.sha256") -Force
            $chk = InUbuntu "cd ~ && sha256sum -c $name.sha256 2>&1 | tail -1"
            if ($chk -notmatch ': OK$') { Warn "the release file does not match its .sha256 ($chk): copy it again"; Done 1 }
            Ok "checksum matches"
        } else { Warn "no .sha256 beside the release file: its integrity is not checked" }
        & wsl.exe -d $Distro -- bash -lc "cd ~ && tar -xzf $name && test -f ~/insar_pipeline/install/install.sh"
        if ($LASTEXITCODE -ne 0) { Warn "unpacking failed"; Done 1 }
        Ok "unpacked into ~/insar_pipeline"
    }
    if ($run) {
        Write-Host "   The installer runs now (about 30 minutes; it asks for the Earthdata login,"
        Write-Host "   the CDS token and the projects folder). Its output follows."
        & wsl.exe -d $Distro -- bash -lc "bash ~/insar_pipeline/install/install.sh"
        if ($LASTEXITCODE -ne 0) {
            Warn "the installer stopped with an error (above). Fix it and run this setup again."
            Done 1
        }
        Ok "installed"
    }
}

# ------------------------------------------------------------ 4. the icon
Say "4. Desktop and Start menu icon"
$src = InUbuntu "test -f ~/insar_pipeline/install/windows/insar.ico && wslpath -w ~/insar_pipeline/install/windows"
if (-not $src) {
    if (Test-Path (Join-Path $Here "insar.ico")) { $src = $Here }
    else { Warn "cannot find install/windows in ~/insar_pipeline"; Done 1 }
}
$dest = Join-Path $env:LOCALAPPDATA "InSAR Pipeline"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item (Join-Path $src "InSAR Pipeline.vbs") $dest -Force
Copy-Item (Join-Path $src "insar.ico") $dest -Force
Set-Content -Path (Join-Path $dest "distro.txt") -Value $Distro -Encoding ASCII
$ws = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
    $lnk = $ws.CreateShortcut((Join-Path $dir "InSAR Pipeline.lnk"))
    $lnk.TargetPath = Join-Path $env:WINDIR "System32\wscript.exe"
    $lnk.Arguments = "`"$(Join-Path $dest 'InSAR Pipeline.vbs')`""
    $lnk.WorkingDirectory = $dest
    $lnk.IconLocation = Join-Path $dest "insar.ico"
    $lnk.Description = "InSAR Pipeline: projects, live runs and results"
    $lnk.Save()
}
Ok "'InSAR Pipeline' is on the desktop and in the Start menu"

if (Ask "Open the InSAR Pipeline now?") { Start-Process (Join-Path $env:WINDIR "System32\wscript.exe") "`"$(Join-Path $dest 'InSAR Pipeline.vbs')`"" }
Done 0
