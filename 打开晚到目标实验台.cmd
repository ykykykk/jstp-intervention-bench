@echo off
chcp 65001 >nul
pushd "%~dp0" || exit /b 1
if not exist ".venv\Scripts\pythonw.exe" (
    echo 请按 README.md 安装项目虚拟环境后，再打开独立实验台。
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -X utf8 -m lategoal.window
popd
