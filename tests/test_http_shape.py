"""Tests for reusable HTTP response-shape pytest source generation."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hive.http_shape import (
    JsonPathPart,
    build_http_shape_test,
    build_json_path_assertions,
    parse_json_path,
)


def _generated_test(source: str):
    namespace: dict[str, object] = {}
    exec(compile(source, "<generated-http-shape-test>", "exec"), namespace)
    tests = [value for name, value in namespace.items() if name.startswith("test_")]
    assert len(tests) == 1
    return tests[0]


def test_build_http_shape_test_uses_route_verb_fixture_and_assertion():
    source = build_http_shape_test(
        "/api/v1/projects",
        "GET",
        {"json_path": "projects[].modules", "must": "non_empty"},
        app_fixture="client",
    )

    assert "def test_http_shape_get_api_v1_projects_projects_modules_non_empty(client):" in source
    assert "response = client.get('/api/v1/projects')" in source
    assert "assert 200 <= response.status_code < 300" in source
    assert "payload = response.json()" in source
    compile(source, "<generated-http-shape-test>", "exec")


def test_json_path_array_field_becomes_per_item_assertion():
    assert parse_json_path("projects[].modules") == (
        JsonPathPart("projects", iterates=True),
        JsonPathPart("modules"),
    )

    source = build_json_path_assertions("projects[].modules", "non_empty")

    assert "assert isinstance(_path_value_0, list)" in source
    assert "assert _path_value_0" in source
    assert "for _path_item_0 in _path_value_0:" in source
    assert "assert 'modules' in _path_item_0" in source
    assert "assert _path_value_1" in source


def test_terminal_array_exists_assertion_is_valid_source():
    source = build_http_shape_test(
        "/api/v1/projects",
        "get",
        {"json_path": "projects[]", "must": "exists"},
        app_fixture="client",
    )

    compile(source, "<generated-http-shape-test>", "exec")
    assert "assert isinstance(_path_value_0, list)" in source
    assert "for _path_item_0" not in source


def test_generated_test_distinguishes_missing_and_present_fields():
    red_app = FastAPI()
    green_app = FastAPI()

    @red_app.get("/api/v1/projects")
    def red_projects():
        return {"projects": [{"project_id": "p1"}]}

    @green_app.get("/api/v1/projects")
    def green_projects():
        return {"projects": [{"project_id": "p1", "modules": ["core"]}]}

    source = build_http_shape_test(
        "/api/v1/projects",
        "get",
        {"json_path": "projects[].modules", "must": "non_empty"},
        app_fixture="client",
    )
    generated_test = _generated_test(source)

    with pytest.raises(AssertionError, match="missing field 'modules'"):
        generated_test(TestClient(red_app))
    generated_test(TestClient(green_app))


@pytest.mark.parametrize(
    ("json_path", "must"),
    [
        ("", "exists"),
        ("projects..modules", "exists"),
        ("projects[0].modules", "exists"),
        ("projects", "truthy"),
    ],
)
def test_invalid_assertion_contract_is_rejected(json_path, must):
    with pytest.raises(ValueError):
        build_json_path_assertions(json_path, must)
