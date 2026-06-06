"""Tests for hive.http_shape_synth — the consumer of GPT's build_http_shape_test
scaffold that recognises the FE-bound-field-over-HTTP symptom (mid-term lever ⑦).

A tiny on-disk codebase (a FastAPI route + a conftest TestClient fixture) drives the
real route grounding (_resolve_http_bindings, mount-prefix folded) so detection,
fixture discovery, and synthesis are all exercised deterministically — no model, no
FlowGate. Negative cases prove the fail-open contract: a missing URL / field / shape /
harness yields None and leaves the caller's behaviour untouched."""
import os

from hive import http_shape_synth as hss


def _make_codebase(tmp_path, *, with_fixture=True, container="projects",
                   field_access="Array.isArray(it.modules)"):
    """A minimal FastAPI app: GET /api/v1/projects returning {container: [...]}."""
    root = tmp_path / "code"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1")\n\n'
        '@router.get("/projects")\n'
        "def list_projects():\n"
        f'    return {{"{container}": _rows()}}\n',
        encoding="utf-8")
    if with_fixture:
        (root / "tests" / "conftest.py").write_text(
            "import pytest\n"
            "from fastapi.testclient import TestClient\n\n"
            "@pytest.fixture\n"
            "def client():\n"
            "    from app.routes import router\n"
            "    from fastapi import FastAPI\n"
            "    app = FastAPI(); app.include_router(router)\n"
            "    return TestClient(app)\n",
            encoding="utf-8")
    return str(root)


def _honey(field_access="Array.isArray(it.modules)"):
    return (
        "Symptom: the module selector never renders.\n"
        'const res = await getRequest("/api/v1/projects")\n'
        "projects.value = list.map((item) => ({\n"
        "  project: it.project,\n"
        f"  modules: {field_access} ? it.modules : [],\n"
        "}))\n")


def test_detect_symptom_resolves_route_field_and_container(tmp_path):
    root = _make_codebase(tmp_path)
    s = hss.detect_http_shape_symptom(_honey(), root)
    assert s is not None
    assert s.verb == "get"
    assert s.full_path == "/api/v1/projects"
    assert s.field == "modules"
    assert s.container == "projects"
    assert s.json_path == "projects[].modules"


def test_detect_returns_none_without_fetch_url(tmp_path):
    root = _make_codebase(tmp_path)
    # No HTTP call literal → not this symptom.
    assert hss.detect_http_shape_symptom(
        "modules: Array.isArray(it.modules) ? it.modules : []", root) is None


def test_detect_returns_none_with_ambiguous_fields(tmp_path):
    root = _make_codebase(tmp_path)
    honey = (_honey() + "\nextra: Array.isArray(it.attachments) ? it.attachments : []\n")
    # Two distinct array fields → ambiguous → declines (never guesses).
    assert hss.detect_http_shape_symptom(honey, root) is None


def test_detect_returns_none_when_no_field(tmp_path):
    root = _make_codebase(tmp_path)
    honey = 'const res = await getRequest("/api/v1/projects")\nconst x = res.data\n'
    assert hss.detect_http_shape_symptom(honey, root) is None


def test_fe_array_fields_only_counts_isarray_shape():
    text = ("Array.isArray(it.modules) ; foo.modules.length ; bar.tags ; "
            "Array.isArray(row.modules)")
    assert hss._fe_array_fields(text) == ["modules"]  # only the isArray shape votes


def test_discover_app_fixture_finds_testclient_fixture(tmp_path):
    root = _make_codebase(tmp_path, with_fixture=True)
    assert hss.discover_app_fixture(root) == "client"


def test_discover_app_fixture_none_when_absent(tmp_path):
    root = _make_codebase(tmp_path, with_fixture=False)
    assert hss.discover_app_fixture(root) is None


def test_synthesize_with_discovered_fixture(tmp_path):
    root = _make_codebase(tmp_path, with_fixture=True)
    out = hss.synthesize_http_shape_red_test(_honey(), root)
    assert out is not None
    edit = out["edit"]
    assert edit["kind"] == "create_file"
    assert edit["file"] == "tests/test_http_shape_modules.py"
    assert out["node"].startswith("tests/test_http_shape_modules.py::")
    # The generated test calls the discovered fixture and the grounded route.
    assert "def client" not in edit["content"]  # relies on the existing conftest fixture
    assert "client.get('/api/v1/projects')" in edit["content"]
    assert "'modules'" in edit["content"]


def test_synthesize_with_explicit_setup_block(tmp_path):
    root = _make_codebase(tmp_path, with_fixture=False)  # no reusable fixture
    setup = ("import pytest\n"
             "from fastapi.testclient import TestClient\n\n"
             "@pytest.fixture\n"
             "def client():\n"
             "    from fastapi import FastAPI\n"
             "    return TestClient(FastAPI())\n")
    out = hss.synthesize_http_shape_red_test(_honey(), root, setup_block=setup)
    assert out is not None
    # The setup block is prepended and the fixture name is read from it.
    assert out["edit"]["content"].startswith("import pytest")
    assert "client.get('/api/v1/projects')" in out["edit"]["content"]


def test_synthesize_fail_open_without_harness(tmp_path):
    root = _make_codebase(tmp_path, with_fixture=False)  # no fixture, no setup_block
    assert hss.synthesize_http_shape_red_test(_honey(), root) is None


def test_synthesize_fail_open_without_symptom(tmp_path):
    root = _make_codebase(tmp_path, with_fixture=True)
    assert hss.synthesize_http_shape_red_test("no symptom here", root) is None
