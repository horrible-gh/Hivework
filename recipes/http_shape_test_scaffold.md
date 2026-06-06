# HTTP response-shape pytest scaffold

`hive.http_shape.build_http_shape_test` generates a focused pytest gate that
calls an endpoint through an existing FastAPI `TestClient` fixture and checks a
JSON path.

```python
from hive.http_shape import build_http_shape_test

source = build_http_shape_test(
    "/api/v1/projects",
    "get",
    {"json_path": "projects[].modules", "must": "non_empty"},
    app_fixture="client",
)
```

Write the returned source under the target repository's `server/tests/` and run
it with `cwd=server`. The generated test expects a fixture named `client`; it
does not start a server, create a database, seed records, or override
dependencies.

## FlowGate fixture pattern

Follow `server/tests/test_outbound_list.py` and
`server/tests/test_settings_api.py`:

1. Set `TESTING=1` before importing FlowGate application modules.
2. Use the existing temporary SQLite migration/seed fixture from the test file
   or `server/tests/conftest.py`.
3. Seed the records needed to make the asserted collection meaningful. For
   `projects[].modules`, seed at least one active project and one corresponding
   `project_modules` row so an empty project list cannot make the check vacuous.
4. Patch FlowGate's existing store/dependencies to that seeded database.
5. Expose the resulting in-process `TestClient` as the fixture passed in
   `app_fixture`.

The client fixture should use the application/router composition whose route
ordering is under test. Reusing the real app composition is important when two
routes can match the same path; mounting only the suspected router would not
exercise route shadowing.

Minimal placement alongside an existing FlowGate client fixture:

```python
# server/tests/test_projects_http_shape.py
# The local or shared `client` fixture owns TestClient + seeded temporary DB.

def test_http_shape_get_api_v1_projects_projects_modules_non_empty(client):
    response = client.get("/api/v1/projects")
    assert 200 <= response.status_code < 300, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    assert "projects" in payload
    projects = payload["projects"]
    assert isinstance(projects, list)
    assert projects
    for project in projects:
        assert isinstance(project, dict)
        assert "modules" in project
        assert project["modules"]
```

Supported assertion modes are `exists` and `non_empty`. JSON paths use dotted
object fields and `[]` for array traversal, for example
`projects[].modules` or `result.items[].name`.
