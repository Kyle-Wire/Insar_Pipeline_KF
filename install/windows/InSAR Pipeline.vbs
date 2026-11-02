' InSAR Pipeline: what the desktop icon runs (install/windows/setup.ps1 puts
' it in %LOCALAPPDATA%\InSAR Pipeline and points the shortcut at it).
'
' The pipeline lives in Ubuntu (WSL). This starts `insar app` there with no
' console window; the app opens its own window (Microsoft Edge in app mode)
' and keeps serving it. A second click while it is running only opens
' another window. Runs are detached: closing the window never stops one.
Option Explicit
Dim sh, distro, cfg, fso
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
distro = "Ubuntu"
' setup.ps1 writes the distribution it installed into beside this file
cfg = fso.BuildPath(fso.GetParentFolderName(WScript.ScriptFullName), "distro.txt")
If fso.FileExists(cfg) Then
  distro = Trim(fso.OpenTextFile(cfg).ReadLine())
End If
' window style 0 = hidden; False = do not wait (the app keeps running)
sh.Run "wsl.exe -d " & distro & " -- bash -lc ""exec ~/.local/bin/insar app""", 0, False
