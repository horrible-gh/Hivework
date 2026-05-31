"""Apply backup store — scratch snapshots that make ``apply --write`` undoable.

``apply --write`` is the one place Hivework mutates a target codebase. Because
Hivework is an independent tool and cannot assume the target lives under git,
every write first snapshots the original files into a scratch *bundle* that sits
OUTSIDE the target tree (so it never clutters the codebase with ``.bak`` files).
Bundles self-expire: each write run first purges bundles older than the TTL, so
the backup store is a time-boxed undo window rather than permanent litter.

Layout (under ``<backup_root>``)::

    <backup_root>/
      20260531T091500Z_NR160_fix/          one bundle per write run
        manifest.json                      created_utc / expires_utc / ttl_hours
                                           / spec / codebase_root / files:[rel,...]
        files/<rel/path/to/file.py>        verbatim pre-write copy (dirs mirrored)

The bundle is the SSOT for an undo: ``restore_bundle`` copies every ``files/<rel>``
back over ``<codebase_root>/<rel>``.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("hive.backup")

MANIFEST_NAME = "manifest.json"
FILES_SUBDIR = "files"
_STAMP_FMT = "%Y%m%dT%H%M%SZ"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sanitize(name: str) -> str:
    """Reduce a spec stem to a filesystem-safe bundle suffix."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return safe.strip("._-") or "spec"


def create_bundle(
    backup_root: str,
    spec_path: str,
    codebase_root: str,
    rel_paths: list[str],
    ttl_hours: int,
    now: datetime | None = None,
    created_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Snapshot ``rel_paths`` (originals) into a fresh bundle; return its record.

    Captures each file's current bytes BEFORE any edit is applied, so a later
    ``restore_bundle`` reproduces the exact pre-write state. Returns a dict with
    ``dir`` (bundle path), ``originals`` (rel -> original text, for in-process
    rollback), and the manifest.

    ``created_paths`` lists files that do not yet exist and will be written by
    this run. They have no bytes to snapshot, so the bundle records them under
    the ``"created"`` manifest key; ``restore_bundle`` deletes them on undo
    rather than rewriting original content.
    """
    now = now or _now()
    stamp = now.strftime(_STAMP_FMT)
    suffix = _sanitize(os.path.splitext(os.path.basename(spec_path))[0])
    bundle_dir = os.path.join(backup_root, f"{stamp}_{suffix}")
    # Extremely unlikely collision (same second, same spec) — disambiguate.
    n = 1
    while os.path.exists(bundle_dir):
        bundle_dir = os.path.join(backup_root, f"{stamp}_{suffix}_{n}")
        n += 1
    files_dir = os.path.join(bundle_dir, FILES_SUBDIR)
    # Create the bundle dir explicitly: a create_file-only run has no rel_paths to
    # snapshot, so the per-file makedirs below never fires and the manifest write
    # would land in a missing directory.
    os.makedirs(bundle_dir, exist_ok=True)

    originals: dict[str, str] = {}
    saved: list[str] = []
    for rel in rel_paths:
        abs_src = os.path.join(codebase_root, rel)
        with open(abs_src, "r", encoding="utf-8") as f:
            text = f.read()
        originals[rel] = text
        abs_dst = os.path.join(files_dir, rel)
        os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
        with open(abs_dst, "w", encoding="utf-8") as f:
            f.write(text)
        saved.append(rel)

    created: list[str] = list(created_paths) if created_paths else []

    expires = now + timedelta(hours=ttl_hours)
    manifest = {
        "created_utc": now.strftime(_STAMP_FMT),
        "expires_utc": expires.strftime(_STAMP_FMT),
        "ttl_hours": ttl_hours,
        "spec": os.path.abspath(spec_path),
        "codebase_root": os.path.abspath(codebase_root),
        "files": saved,
        "created": created,   # paths to DELETE on restore (no original bytes)
    }
    with open(os.path.join(bundle_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Backup bundle created: %s (%d modified, %d created, expires %s)",
                bundle_dir, len(saved), len(created), manifest["expires_utc"])
    return {"dir": bundle_dir, "originals": originals, "manifest": manifest}


def load_manifest(bundle_dir: str) -> dict[str, Any]:
    """Load a bundle's manifest.json (raises if missing/corrupt)."""
    with open(os.path.join(bundle_dir, MANIFEST_NAME), "r", encoding="utf-8") as f:
        return json.load(f)


def restore_bundle(bundle_dir: str) -> list[str]:
    """Copy every saved file back over the codebase; delete any created files.

    Modified files (``manifest["files"]``) are restored by overwrite. Created
    files (``manifest["created"]``) had no original bytes, so they are restored
    by deletion; a missing created file is skipped (idempotent). Empty parent
    dirs are not pruned, matching the modified-file behaviour.
    """
    manifest = load_manifest(bundle_dir)
    root = manifest["codebase_root"]
    files_dir = os.path.join(bundle_dir, FILES_SUBDIR)
    restored: list[str] = []
    for rel in manifest.get("files", []):
        abs_src = os.path.join(files_dir, rel)
        abs_dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
        shutil.copyfile(abs_src, abs_dst)
        restored.append(abs_dst)
    for rel in manifest.get("created", []):
        abs_dst = os.path.join(root, rel)
        try:
            os.remove(abs_dst)
            restored.append(abs_dst)
        except FileNotFoundError:
            logger.warning("restore: created file already absent, skipping: %s", rel)
        except OSError as exc:
            logger.error("restore: could not delete created file %s: %s", rel, exc)
    logger.info("Restored %d file(s) from %s", len(restored), bundle_dir)
    return restored


def latest_bundle(backup_root: str) -> str | None:
    """Return the most recently created bundle dir, or None if the store is empty."""
    if not os.path.isdir(backup_root):
        return None
    candidates = [
        os.path.join(backup_root, name)
        for name in os.listdir(backup_root)
        if os.path.isfile(os.path.join(backup_root, name, MANIFEST_NAME))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: os.path.getmtime(d))


def purge_expired(
    backup_root: str,
    now: datetime | None = None,
    fallback_ttl_hours: int = 168,
) -> list[str]:
    """Delete bundles whose ``expires_utc`` has passed; return removed dir names.

    A bundle's own manifest carries its expiry (TTL can differ per run). If the
    manifest is missing or unreadable, the bundle is purged on a conservative
    fallback: directory mtime older than ``fallback_ttl_hours``.
    """
    now = now or _now()
    if not os.path.isdir(backup_root):
        return []

    removed: list[str] = []
    for name in os.listdir(backup_root):
        bundle_dir = os.path.join(backup_root, name)
        if not os.path.isdir(bundle_dir):
            continue
        expired = False
        try:
            manifest = load_manifest(bundle_dir)
            expires = datetime.strptime(
                manifest["expires_utc"], _STAMP_FMT).replace(tzinfo=timezone.utc)
            expired = now >= expires
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            mtime = datetime.fromtimestamp(
                os.path.getmtime(bundle_dir), tz=timezone.utc)
            expired = now >= mtime + timedelta(hours=fallback_ttl_hours)
        if expired:
            shutil.rmtree(bundle_dir, ignore_errors=True)
            removed.append(name)

    if removed:
        logger.info("Purged %d expired backup bundle(s): %s",
                    len(removed), ", ".join(removed))
    return removed
