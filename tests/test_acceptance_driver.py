"""Tests for the box-0 new-feature driver (decompose §2.6) and the specify wiring pass.

The driver tests prove recipe SELECTION (operator override > feature > bug default) and
the acceptance-axis ORDER enforcement (feature recipe forces the axis to the front of
step 0; bug recipe is untouched). The wiring test proves the specify pass attaches a
box-0 red-test node to a spec carrying a source edit when criteria + grounding resolve,
and is a no-op (fail-open) when criteria text is absent — the operational handoff is
supplied from outside the chain (NR0003 conclusion / L DEFERRED)."""

from hive import decompose as dec
from hive import specify


# ── decompose §2.6 driver ────────────────────────────────────────────────────
def test_is_new_feature_request_bug_signal_vetoes():
    assert dec.is_new_feature_request("새 기능 추가: 댓글 작성 화면") is True
    assert dec.is_new_feature_request("implement the export endpoint") is True
    # a bug signal vetoes even if a feature word is present → safe default (bug)
    assert dec.is_new_feature_request("implement fix for the 500 error") is False
    assert dec.is_new_feature_request("로그인 버튼이 안 됨") is False


def test_select_recipe_override_feature_bug():
    assert dec.select_recipe("anything", explicit_recipe="recipe_x") == "recipe_x"
    assert dec.select_recipe("새 기능 추가") == dec.FEATURE_RECIPE_ID
    assert dec.select_recipe("500 error on save") == "recipe_code_bug"


def test_enforce_feature_axis_order_prepends_for_feature():
    result = {"steps": [["A", "B"]],
              "tasks": [{"id": "A"}, {"id": "B"}]}
    out = dec.enforce_feature_axis_order(result, dec.FEATURE_RECIPE_ID)
    assert out["tasks"][0]["id"] == dec.ACCEPTANCE_AXIS_ID  # axis at front
    assert out["steps"][0][0] == dec.ACCEPTANCE_AXIS_ID     # head of step 0


def test_enforce_feature_axis_order_noop_for_bug_recipe():
    result = {"steps": [["A"]], "tasks": [{"id": "A"}]}
    out = dec.enforce_feature_axis_order(result, "recipe_code_bug")
    assert [t["id"] for t in out["tasks"]] == ["A"]  # untouched


def test_enforce_feature_axis_order_idempotent():
    result = {"steps": [["A"]], "tasks": [{"id": "A"}]}
    once = dec.enforce_feature_axis_order(result, dec.FEATURE_RECIPE_ID)
    twice = dec.enforce_feature_axis_order(once, dec.FEATURE_RECIPE_ID)
    axis_count = sum(1 for t in twice["tasks"] if t["id"] == dec.ACCEPTANCE_AXIS_ID)
    assert axis_count == 1  # no double-insert


# ── specify wiring pass ──────────────────────────────────────────────────────
def _codebase(tmp_path):
    root = tmp_path / "code"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1")\n\n'
        '@router.get("/projects")\n'
        "def list_projects():\n"
        '    return {"projects": _rows()}\n', encoding="utf-8")
    (root / "tests" / "conftest.py").write_text(
        "import pytest\n"
        "from fastapi.testclient import TestClient\n\n"
        "@pytest.fixture\n"
        "def client():\n"
        "    from app.routes import router\n"
        "    from fastapi import FastAPI\n"
        "    app = FastAPI(); app.include_router(router)\n"
        "    return TestClient(app)\n", encoding="utf-8")
    return str(root)


_CRITERIA = (
    "## 수용기준\n"
    "- id: AC1\n"
    "  prose: GET /api/v1/projects must return a non-empty `modules` field\n")


def test_specify_pass_attaches_node_for_resolving_criterion(tmp_path):
    root = _codebase(tmp_path)
    spec = {"status": "ready_to_apply",
            "edits": [{"id": "FIX", "kind": "edit", "file": "app/routes.py",
                       "anchor_old": "x", "replacement_new": "y"}]}
    out = specify._synthesize_acceptance_red_test(spec, _CRITERIA, root)
    assert out["verify"]["red_test_node"].startswith("tests/test_acceptance_modules.py::")
    assert any(e["id"] == "ACCEPTANCE_RED" for e in out["edits"])


def test_specify_pass_noop_without_criteria_text(tmp_path):
    root = _codebase(tmp_path)
    spec = {"edits": [{"id": "FIX", "kind": "edit", "file": "app/routes.py"}]}
    out = specify._synthesize_acceptance_red_test(spec, None, root)
    assert "verify" not in out or not out.get("verify", {}).get("red_test_node")


def test_specify_pass_noop_without_source_edit(tmp_path):
    root = _codebase(tmp_path)
    # only a test edit → nothing to certify → no synthesis
    spec = {"edits": [{"id": "T", "kind": "create_file", "file": "tests/test_x.py"}]}
    out = specify._synthesize_acceptance_red_test(spec, _CRITERIA, root)
    assert not out.get("verify", {}).get("red_test_node")


def test_specify_pass_does_not_clobber_existing_node(tmp_path):
    root = _codebase(tmp_path)
    spec = {"edits": [{"id": "FIX", "kind": "edit", "file": "app/routes.py"}],
            "verify": {"red_test_node": "tests/existing.py::test_keep"}}
    out = specify._synthesize_acceptance_red_test(spec, _CRITERIA, root)
    assert out["verify"]["red_test_node"] == "tests/existing.py::test_keep"
