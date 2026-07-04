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

# Pick the host interpreter: 'python3' on *nix, 'python' on Windows git-bash.
# Each candidate must actually RUN, not just resolve — 'command -v' alone would
# accept the fake Windows-Store 'python3' alias stub, which only opens the
# Store and exits nonzero.
PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import sys" >/dev/null 2>&1; then
    PY="$cand"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3 not found on PATH (tried: python3, python)." >&2
  echo "  Debian/Ubuntu:      sudo apt install python3 python3-venv" >&2
  echo "  Windows (git-bash): install Python from https://www.python.org/downloads/" >&2
  exit 1
fi

# On Debian/Ubuntu without python3-venv, 'python -m venv' fails AFTER creating a
# pip-less half-venv; drop the debris so a re-run retries cleanly instead of
# skipping creation and dying later at 'pip install' with "No module named pip".
create_venv() {
  echo "Creating virtual environment .venv ..."
  if ! "$PY" -m venv .venv; then
    rm -rf .venv
    echo "" >&2
    echo "venv creation failed. On Debian/Ubuntu install the venv module first:" >&2
    echo "  sudo apt install python3-venv" >&2
    exit 1
  fi
}

# venv layout differs by platform: Scripts/ on Windows (git-bash), bin/ elsewhere.
if [ ! -x ".venv/Scripts/python.exe" ] && [ ! -x ".venv/bin/python" ]; then
  create_venv
fi
if [ -x ".venv/Scripts/python.exe" ]; then
  VPY=".venv/Scripts/python.exe"
else
  VPY=".venv/bin/python"
fi

# A .venv left by an earlier failed create can have a python but no pip.
# Success means "pip answers", not "the interpreter file exists" — rebuild it.
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
  echo "Existing .venv has no working pip (earlier create failed halfway) - recreating ..."
  rm -rf .venv
  create_venv
  if [ -x ".venv/Scripts/python.exe" ]; then
    VPY=".venv/Scripts/python.exe"
  else
    VPY=".venv/bin/python"
  fi
fi

echo "Installing dependencies into .venv ..."
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt

"$VPY" hive_setup.py
