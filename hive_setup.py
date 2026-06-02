#!/usr/bin/env python3
"""Hivework interactive setup.

Run via setup.bat / setup.sh (which install requirements first), or directly:

    python hive_setup.py

Three idempotent steps - nothing already present is ever overwritten:

  1. Bootstrap ``hive.config.json`` from ``hive.config.example.json`` (repo-local,
     gitignored - fill in each role's provider/model and your db_connections).
  2. Create the out-of-repo secrets file ``~/.hivework/.env`` (or ``$HIVE_ENV_FILE``)
     so API keys never live inside the repo, and optionally store DEEPINFRA_TOKEN now.
  3. Print next steps.

UI is English only (this runs once; a language layer can be added later if wanted).
Uses ``rich`` for clean output when available, and degrades to plain text if not.
Console output is kept ASCII-only so it cannot crash on a legacy Windows code page
(cp932/cp949), the same hazard hive.py guards against.
"""
from __future__ import annotations

import os
import sys


def _force_utf8_io() -> None:
    """Best-effort UTF-8 stdout/stderr so output never dies on a legacy code page."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8_io()

REPO = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(REPO, "hive.config.json")
CONFIG_EXAMPLE = os.path.join(REPO, "hive.config.example.json")

# ── UI shim: prefer rich, fall back to plain stdio so a deps-less direct run still
#    works. All text is ASCII so neither path can hit a code-page encode error.
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    _con = Console()

    def banner(text: str) -> None:
        _con.print(Panel(text, expand=False))

    def info(text: str) -> None:
        _con.print(text)

    def ok(text: str) -> None:
        _con.print(f"[green][ok][/green] {text}")

    def warn(text: str) -> None:
        _con.print(f"[yellow][!][/yellow] {text}")

    def ask_secret(prompt: str) -> str:
        return Prompt.ask(prompt, password=True, default="", show_default=False)

except Exception:  # rich not installed (direct run before pip install) - plain stdio
    import getpass

    def banner(text: str) -> None:
        line = "=" * min(len(text) + 4, 72)
        print(f"\n{line}\n  {text}\n{line}")

    def info(text: str) -> None:
        print(text)

    def ok(text: str) -> None:
        print(f"[ok] {text}")

    def warn(text: str) -> None:
        print(f"[!] {text}")

    def ask_secret(prompt: str) -> str:
        try:
            return getpass.getpass(prompt + " ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""


def _secrets_path() -> str:
    """Where the secrets file lives: $HIVE_ENV_FILE override, else ~/.hivework/.env."""
    return os.environ.get("HIVE_ENV_FILE") or os.path.join(
        os.path.expanduser("~"), ".hivework", ".env")


def _secrets_skeleton(token: str = "") -> str:
    return (
        "# Hivework secrets - keep this file OUTSIDE the repo (it holds real keys).\n"
        "# A value already in your environment wins; this file only fills gaps.\n"
        f"DEEPINFRA_TOKEN={token}\n"
        "# DB passwords referenced by hive.config.json db_connections[*].password_env\n"
        "# (only for mysql/postgres targets; sqlite needs none), e.g.:\n"
        "# DB_PASSWORD=\n"
    )


def bootstrap_config() -> None:
    """Create hive.config.json from the example, unless it already exists."""
    if os.path.isfile(CONFIG):
        ok(f"Config already exists - left untouched:\n    {CONFIG}")
        return
    if not os.path.isfile(CONFIG_EXAMPLE):
        warn("hive.config.example.json not found - skipping config bootstrap.")
        return
    import shutil
    shutil.copyfile(CONFIG_EXAMPLE, CONFIG)
    ok(f"Created config from the example:\n    {CONFIG}")
    info("    -> Edit it: set each role's provider/model, and your db_connections.")


def bootstrap_secrets() -> None:
    """Create ~/.hivework/.env (or $HIVE_ENV_FILE), optionally storing DEEPINFRA_TOKEN."""
    path = _secrets_path()
    if os.path.isfile(path):
        ok(f"Secrets file already exists - left untouched:\n    {path}")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    info("\nDEEPINFRA_TOKEN is the API key for the judge / converge / review roles.")
    token = ask_secret("Paste it now (or leave blank to fill in later):")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_secrets_skeleton(token.strip()))
    if token.strip():
        ok(f"Saved DEEPINFRA_TOKEN to:\n    {path}")
    else:
        ok(f"Created secrets file (token left blank):\n    {path}")
        info("    -> Open it later and set DEEPINFRA_TOKEN before a paid run.")


def main() -> None:
    banner("Hivework setup")
    info("Idempotent - anything already present is left untouched.\n")
    bootstrap_config()
    info("")
    bootstrap_secrets()
    info("")
    banner("Setup complete")
    info("Next:")
    info("  - Standalone runs read ~/.hivework/.env automatically.")
    info("  - Via the launcher, the launcher's environment wins - the file isn't required.")
    info("  - Point $HIVE_ENV_FILE at another path to relocate the secrets file.")


if __name__ == "__main__":
    main()
