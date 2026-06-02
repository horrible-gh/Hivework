@echo off
REM ---------------------------------------------------------------------------
REM Hivework one-shot setup (Windows):
REM   1. create a project-local virtual environment (.venv) if missing,
REM   2. install dependencies INTO that .venv (never the global Python),
REM   3. run the interactive setup (config bootstrap + out-of-repo secrets).
REM   Usage:  setup.bat
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "VPY=.venv\Scripts\python.exe"

if not exist "%VPY%" (
  echo Creating virtual environment .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo.
    echo venv creation failed. Is Python on your PATH? Try:  py -m venv .venv
    exit /b 1
  )
)

echo Installing dependencies into .venv ...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Dependency install failed.
  exit /b 1
)

"%VPY%" hive_setup.py
endlocal
