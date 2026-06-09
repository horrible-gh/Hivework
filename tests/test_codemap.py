"""Acceptance tests for the deterministic static code map."""

from pathlib import Path

import pytest

from hive.codemap import (
    endpoints_in_file,
    field_producers,
    find_symbol,
    handler_callees,
    route_table,
)


M035 = Path(r"C:\workspace\projects\FlowGate-dev\branches\20260608_M035")
M036 = Path(r"C:\workspace\projects\FlowGate-dev\branches\20260608_M036")
M037 = Path(r"C:\workspace\projects\FlowGate-dev\branches\20260607")


def _root(path: Path) -> str:
    if not path.is_dir():
        pytest.skip(f"FlowGate acceptance worktree is unavailable: {path}")
    return str(path)


def test_route_table_marks_shadowed_projects_handler():
    routes = [
        row for row in route_table(_root(M036))
        if row["path"] == "/api/v1/projects" and "GET" in row["methods"]
    ]

    project_settings = next(
        row for row in routes if row["handler_file"].endswith(
            "/settings/routers/project_settings.py"
        )
    )
    legacy = next(
        row for row in routes if row["handler_file"].endswith(
            "/api/v1/legacy_misc_routes.py"
        )
    )

    assert project_settings["handler_func"] == "list_projects_endpoint"
    assert project_settings["live"] is True
    assert legacy["live"] is False
    assert project_settings["register_index"] < legacy["register_index"]


def test_frontend_endpoint_reaches_live_handler_and_service_callee():
    root = _root(M037)
    endpoint = "/api/v1/outbox/create"
    assert endpoint in endpoints_in_file(
        root, "client/src/main/components/NewRequirementModal.vue"
    )

    handler = next(
        row for row in route_table(root)
        if row["path"] == endpoint and "POST" in row["methods"] and row["live"]
    )
    assert handler["handler_file"].endswith("/api/v1/legacy_misc_routes.py")
    assert handler["handler_func"] == "api_outbox_create"

    callees = handler_callees(
        root, handler["handler_file"], handler["handler_func"]
    )
    assert {
        (row["name"], row["module_file"]) for row in callees
    } >= {
        (
            "process_service.create_requirement",
            "server/modules/flow_gate/process_service.py",
        )
    }


def test_field_producers_reaches_workflow_head_parser():
    producers = field_producers(_root(M035), "workflow_head_type")

    assert any(
        row["file"] == "server/modules/flow_gate/documents/routers/documents.py"
        for row in producers
    )


def test_find_symbol_reaches_process_service_definition():
    definitions = find_symbol(_root(M037), "create_requirement")

    assert any(
        row["file"] == "server/modules/flow_gate/process_service.py"
        and row["kind"] == "function"
        for row in definitions
    )


def test_route_table_walk_fallback_without_git(tmp_path):
    routes_dir = tmp_path / "server" / "api"
    routes_dir.mkdir(parents=True)
    (tmp_path / "server" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from api.routes import router\n"
        "app = FastAPI()\n"
        '@app.get("/health")\n'
        "def health():\n"
        "    return {}\n"
        'app.include_router(router, prefix="/api")\n',
        encoding="utf-8",
    )
    (routes_dir / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/v1")\n'
        '@router.get("/things")\n'
        "def list_things():\n"
        "    return []\n",
        encoding="utf-8",
    )

    assert route_table(str(tmp_path)) == [
        {
            "path": "/health",
            "methods": ["GET"],
            "handler_file": "server/main.py",
            "handler_func": "health",
            "live": True,
            "register_index": 0,
        },
        {
            "path": "/api/v1/things",
            "methods": ["GET"],
            "handler_file": "server/api/routes.py",
            "handler_func": "list_things",
            "live": True,
            "register_index": 1,
        },
    ]
