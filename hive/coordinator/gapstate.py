"""DB-01 gap-state persistence — uuid-keyed transcript + gaps + seed_draft, with
``.apply_backups``-style TTL expiry. Reuses the sqlite/ledger persistence pattern
(its own table, default alongside ``hive_ledger.db``).

In W1 the coordinator is a single-pass 1-shot, so this is mostly audit + a forward
hook for W2 resume (P-01 ``--resume <uuid>``): a loaded-then-expired session
returns None exactly like a missing one, which is the resume failure path P-01 §7
specifies. Non-fatal: any storage error degrades to no-op (the run still hands off).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from hive.coordinator.model import GapState

_DDL = """
CREATE TABLE IF NOT EXISTS coordinator_sessions (
    uuid       TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status     TEXT,
    data       TEXT NOT NULL
);
"""


class GapStateStore:
    """Thin sqlite store for coordinator gap-states. TTL in hours (default 7 days,
    matching apply backup retention)."""

    def __init__(self, db_path: str = "hive_ledger.db", ttl_hours: int = 168):
        self.ttl_hours = ttl_hours
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(_DDL)
        self._conn.commit()

    def save(self, gap_state: GapState) -> None:
        now = _now_iso()
        payload = json.dumps(gap_state.to_dict(), ensure_ascii=False)
        # Preserve created_at on update; set it on first insert (UPSERT).
        self._conn.execute(
            """INSERT INTO coordinator_sessions(uuid, created_at, updated_at, status, data)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(uuid) DO UPDATE SET
                   updated_at=excluded.updated_at,
                   status=excluded.status,
                   data=excluded.data""",
            (gap_state.uuid, now, now, gap_state.status, payload),
        )
        self._conn.commit()

    def load(self, uuid: str) -> GapState | None:
        """Return the session, or None if missing OR past its TTL (P-01 §7:
        expired uuid load fails → caller restarts a fresh session)."""
        row = self._conn.execute(
            "SELECT created_at, data FROM coordinator_sessions WHERE uuid=?",
            (uuid,),
        ).fetchone()
        if row is None:
            return None
        created_at, data = row
        if self._expired(created_at):
            return None
        try:
            return GapState.from_dict(json.loads(data))
        except (ValueError, KeyError):
            return None

    def purge_expired(self) -> int:
        cutoff = (_now() - timedelta(hours=self.ttl_hours)).isoformat()
        cur = self._conn.execute(
            "DELETE FROM coordinator_sessions WHERE created_at < ?", (cutoff,))
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def _expired(self, created_at: str) -> bool:
        try:
            born = datetime.fromisoformat(created_at)
        except ValueError:
            return False
        if born.tzinfo is None:
            born = born.replace(tzinfo=timezone.utc)
        return _now() - born > timedelta(hours=self.ttl_hours)


def open_store(db_path: str = "hive_ledger.db", ttl_hours: int = 168,
               enabled: bool = True) -> "GapStateStore | None":
    """Factory: returns a store, or None when disabled / on any open error
    (non-fatal — persistence is best-effort, the run proceeds without it)."""
    if not enabled:
        return None
    try:
        return GapStateStore(db_path, ttl_hours)
    except sqlite3.Error:
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()
