"""Canonical loader for Hivework's out-of-repo secrets file.

Both the CLI (``hive.py``) and the test suite (``tests/conftest.py``) call this so a
token configured ONCE by ``hive_setup`` (``~/.hivework/.env``) reaches every entry
point — no launcher batch / manual ``set`` required. Keys live OUTSIDE the repo:
shipping a key inside a publishable project violates "don't ship secrets with the
code", which is why setup writes the home-dir file rather than the config.

Resolution order:
    1. ``HIVE_ENV_FILE`` if set (point it anywhere — e.g. a shared location), else
    2. ``~/.hivework/.env``  (``%USERPROFILE%\\.hivework\\.env`` on Windows — the home
       dir, NOT AppData).

Format is plain ``KEY=VALUE`` lines (``#`` comments and blank lines ignored; optional
surrounding quotes stripped). Precedence is ``setdefault`` — a value already in the
real environment (e.g. injected by a launcher that runs us, or by CI) WINS over the
file, so launcher-driven, standalone, and pytest runs all work without clobbering an
explicit override. Never raises: a missing/garbled file just leaves the env unchanged
and a downstream provider reports its own missing-key error.
"""
from __future__ import annotations

import os


def secrets_path() -> str:
    """Resolve the secrets file path (``$HIVE_ENV_FILE`` override, else ``~/.hivework/.env``)."""
    return os.environ.get("HIVE_ENV_FILE") or os.path.join(
        os.path.expanduser("~"), ".hivework", ".env")


def load_secrets() -> str | None:
    """Load the secrets file into ``os.environ`` (real env wins). Return the path loaded, or None."""
    path = secrets_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)  # real env wins; file only fills gaps
    return path
