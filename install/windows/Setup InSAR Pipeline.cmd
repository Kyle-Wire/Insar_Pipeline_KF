@echo off
rem Double-click to set up the InSAR pipeline on this PC (setup.ps1 beside this file).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
