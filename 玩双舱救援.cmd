@echo off
setlocal
pushd "%~dp0" || exit /b 1
if exist ".venv\Scripts\python.exe" goto play
echo Project Python not found: "%CD%\.venv\Scripts\python.exe"
set "rc=1"
goto finished

:play
".venv\Scripts\python.exe" -X utf8 -m jstp.rescue_game
set "rc=%errorlevel%"

:finished
popd
if "%rc%"=="0" exit /b 0
echo.
echo Double Cabin Rescue stopped (exit code %rc%). See the error above.
pause
exit /b %rc%
