@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "py=.venv\Scripts\python.exe"
if exist "work\vla-env\Scripts\python.exe" set "py=work\vla-env\Scripts\python.exe"
if not exist "%py%" goto missing
set "HF_HUB_OFFLINE=1"
"%py%" -X utf8 -m studies.joint_primer.game
set "rc=%errorlevel%"
goto finished

:missing
echo Project Python was not found. Open docs\FOUR-TRACK-GAME.md for the local requirements.
set "rc=1"

:finished
popd
if "%rc%"=="0" exit /b 0
echo.
echo Four-Track Dispatch stopped (exit code %rc%). See the error above.
pause
exit /b %rc%
