# FlowGate write-sink red-test harness (lever L2 setup_block).
#
# Prepended by hive.write_sink_synth to the synthesised not-500 red test for
# POST /api/v1/groups/{group_id}/dispose. It builds a TestClient over FlowGate's REAL
# dispose route (tree_routes.group_dispose -> process_service.dispose_group ->
# db.insert_event / insert_group_event) backed by a freshly-migrated temp SQLite DB with
# foreign_keys ON and ONE seeded OPEN group.
#
# That FK enforcement + seeded group is what makes the assertion BITE: with the pre-fix
# sink (db.insert_event(group_id, ...)) the group-level event is written into
# events.doc_id (NOT NULL REFERENCES documents(doc_id)); a group_id is never a doc_id, so
# the write raises FOREIGN KEY constraint failed -> HTTP 500 (RED). With the callee-swap
# fix (db.insert_group_event(group_id, ...) -> group_events.group_id REFERENCES groups)
# the dispose succeeds -> 200 (GREEN). No mocks: the real store hits the seeded file.
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ["TESTING"] = "1"
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only-32c")

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))
_MIGRATIONS_DIR = _SERVER_DIR / "sql" / "migrations" / "sqlite"

# The seeded group id; the synthesised request path must dispose THIS id.
WRITE_SINK_GROUP_ID = "flowgate.default.0082"


@pytest.fixture
def client():
    from modules.flow_gate.db import _SqliteDbAdapter
    from modules.flow_gate.db import connection as conn_mod
    from modules.flow_gate.db import groups as db_groups
    from modules.flow_gate.db import projects as db_projects
    from modules.flow_gate.api.v1.tree_routes import router as _tree_router

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(db_path)
    for migration in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        try:
            conn.executescript(migration.read_text(encoding="utf-8"))
        except sqlite3.OperationalError:
            pass  # idempotent re-applies / dialect quirks: mirror conftest tolerance
    conn.close()

    original = conn_mod.STORE
    store = conn_mod.FlowGateStore()
    store._db = _SqliteDbAdapter(db_path)
    conn_mod.STORE = store

    db_projects.create({"project_id": "flowgate", "project_name": "flowgate"})
    db_groups.create({
        "group_id": WRITE_SINK_GROUP_ID, "project_id": "flowgate",
        "module": "default", "title": "Dispose target", "status": "OPEN",
    })

    app = FastAPI()
    app.include_router(_tree_router)
    # raise_server_exceptions=False so a handler 500 is observed as a 500 RESPONSE (what the
    # synthesised assertion checks) rather than re-raised into the test body.
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        conn_mod.STORE = original
        try:
            os.unlink(db_path)
        except OSError:
            pass
