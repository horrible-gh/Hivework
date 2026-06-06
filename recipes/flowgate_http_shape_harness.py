# FlowGate HTTP-shape red-test harness (lever ⑦ setup_block).
#
# Prepended by hive.http_shape_synth to the synthesised GET /api/v1/projects red
# test. It builds a TestClient over FlowGate's REAL projects endpoint
# (legacy_misc_routes.api_projects → process_service.get_projects_with_modules →
# db.get_allowed_projects) backed by a freshly-migrated temp SQLite DB seeded with
# one active project AND one groups row carrying a real module value.
#
# That seeding is what makes the assertion BITE: the real module data exists, so a
# correct producer MUST surface it as a non-empty projects[].modules. The current
# bug (store.get_allowed_projects hardcodes "'' AS module") returns it empty, so the
# generated test is RED until the producer is fixed — exactly the red→green the
# apply stage observes. No mocks: the real store reads the seeded file.
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("TESTING", "1")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only-32c")

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))
_MIGRATIONS_DIR = _SERVER_DIR / "sql" / "migrations" / "sqlite"


@pytest.fixture
def client():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    for migration in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        try:
            conn.executescript(migration.read_text(encoding="utf-8"))
        except sqlite3.OperationalError:
            pass  # idempotent re-applies / dialect quirks: mirror conftest tolerance
    # One active project + one groups row whose module is the data a correct
    # producer must surface. With the current bug modules comes back [] (RED);
    # once the producer reads the real module it becomes ["billing"] (GREEN).
    conn.executescript(
        """
        INSERT OR IGNORE INTO projects(project_id,project_name,is_active,created_at,updated_at)
            VALUES('proj_hs','HttpShapeProj',1,datetime('now'),datetime('now'));
        INSERT OR IGNORE INTO groups(group_id,project_id,module,title,status,created_at,updated_at)
            VALUES('grp_hs','proj_hs','billing','G1','OPEN',datetime('now'),datetime('now'));
        -- Isolate the asserted world: the migration seeds a module-less __SYSTEM__
        -- project, and the generated gate asserts EVERY returned project has non-empty
        -- modules. Leaving a legitimately module-less project active would keep the
        -- test RED even after a correct producer fix (a false still_red that over-blocks).
        UPDATE projects SET is_active = 0 WHERE project_id <> 'proj_hs';
        """
    )
    conn.commit()
    conn.close()

    from modules.flow_gate import db as _db
    from modules.flow_gate.store import FlowGateStore
    from modules.flow_gate.api.v1.legacy_misc_routes import router as _legacy_router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    prev_store = _db._store
    _db._store = FlowGateStore(db_path=db_path)
    app = FastAPI()
    app.include_router(_legacy_router)
    try:
        yield TestClient(app)
    finally:
        _db._store = prev_store
        try:
            os.unlink(db_path)
        except OSError:
            pass
