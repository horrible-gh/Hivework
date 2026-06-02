#!/usr/bin/env sh
# ---------------------------------------------------------------------------
# Hivework one-shot setup (*nix / git-bash):
#   1. create a project-local virtual environment (.venv) if missing,
#   2. install dependencies INTO that .venv (never the global Python),
#   3. run the interactive setup (config bootstrap + out-of-repo secrets).
#   Usage:  sh setup.sh
# ---------------------------------------------------------------------------
set -eu
cd "$(dirname "$0")"

# venv layout differs by platform: Scripts/ on Windows (git-bash), bin/ elsewhere.
if [ ! -x ".venv/Scripts/python.exe" ] && [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment .venv ..."
  python -m venv .venv
fi
if [ -x ".venv/Scripts/python.exe" ]; then
  VPY=".venv/Scripts/python.exe"
else
  VPY=".venv/bin/python"
fi

echo "Installing dependencies into .venv ..."
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt

"$VPY" hive_setup.py
