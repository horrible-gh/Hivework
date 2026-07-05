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

# ---------------------------------------------------------------------------
# Locate a usable Python 3 interpreter.
# Linux ships `python3` (Debian/Ubuntu has no bare `python`); Windows/git-bash
# uses `python`. Try python3 first, then python, and *execute* each candidate so
# a non-runnable stub (e.g. the Windows Store `python3` alias that only opens the
# Store) is rejected rather than picked.
# ---------------------------------------------------------------------------
PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import sys" >/dev/null 2>&1; then
    PY="$cand"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "Python 3 not found on PATH (tried: python3, python)." >&2
  echo "  Debian/Ubuntu:       sudo apt install python3 python3-venv" >&2
  echo "  Windows (git-bash):  install Python 3 from https://python.org and reopen the shell" >&2
  exit 1
fi

# venv layout differs by platform: Scripts/ on Windows (git-bash), bin/ elsewhere.
venv_python() {
  if [ -x ".venv/Scripts/python.exe" ]; then
    printf '%s\n' ".venv/Scripts/python.exe"
  else
    printf '%s\n' ".venv/bin/python"
  fi
}

create_venv() {
  echo "Creating virtual environment .venv ..."
  if ! "$PY" -m venv .venv; then
    # A failed `venv` often leaves a half-built .venv behind; clear it so the
    # next run starts clean instead of tripping over the debris.
    rm -rf .venv
    echo "Failed to create the virtual environment." >&2
    echo "  The 'venv' module may be missing. Debian/Ubuntu:  sudo apt install python3-venv" >&2
    exit 1
  fi
}

if [ ! -x ".venv/Scripts/python.exe" ] && [ ! -x ".venv/bin/python" ]; then
  create_venv
fi

VPY="$(venv_python)"

# A previous failed run (typically missing python3-venv) can leave a .venv whose
# interpreter exists but has no pip ("No module named pip"). Detect that and
# rebuild the environment automatically instead of failing downstream.
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
  echo "Existing .venv has no working pip - recreating it ..."
  rm -rf .venv
  create_venv
  VPY="$(venv_python)"
  if ! "$VPY" -m pip --version >/dev/null 2>&1; then
    echo "The recreated .venv still has no working pip." >&2
    echo "  Install the venv module and retry. Debian/Ubuntu:  sudo apt install python3-venv" >&2
    exit 1
  fi
fi

echo "Installing dependencies into .venv ..."
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt

"$VPY" hive_setup.py
