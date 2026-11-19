@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m studies.partial_futures.window
if errorlevel 1 pause
