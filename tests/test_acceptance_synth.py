"""Tests for hive.acceptance_synth — box-0 (group 0064): design ``## 수용기준`` →
acceptance RED test.

A tiny on-disk FastAPI codebase drives the SAME route grounding lever ⑦ uses, so
explicit-oracle validation, prose derivation, the ``equals`` extension, unit_value, and
synthesis are exercised deterministically — no model, no FlowGate. The negative cases
are load-bearing: they prove the fail-open contract (ambiguous route/field, no harness,
no literal, no marker → ``None``/``[]`` and the caller's behaviour is untouched), which
is the whole trust basis of box-0 (a decline is a no-go, never a guessed test).
"""
import textwrap

from hive import acceptance_synth as acc


def _make_codebase(tmp_path, *, with_fixture=True, container="projects"):
    """GET /api/v1/projects returning {container: [...]} + a conftest TestClient."""
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


# ── §2.1 read_acceptance_criteria ────────────────────────────────────────────
def test_read_criteria_extracts_marker_list():
    design = textwrap.dedent("""\
        # Design

        ## 수용기준
        - id: AC1
          prose: the `modules` field on GET /api/v1/projects must be non-empty
        - id: AC2
          prose: status equals done

        ## Other section
        - ignored: true
        """)
    crit = acc.read_acceptance_criteria(design)
    assert [c["id"] for c in crit] == ["AC1", "AC2"]


def test_read_criteria_no_marker_is_empty():
    assert acc.read_acceptance_criteria("# Design\n\nno criteria here") == []


def test_read_criteria_drops_items_without_id_or_prose():
    design = "## 수용기준\n- id: AC1\n- prose: orphan\n- id: AC2\n  prose: ok\n"
    crit = acc.read_acceptance_criteria(design)
    assert [c["id"] for c in crit] == ["AC2"]


# ── §4.2 infer_must ──────────────────────────────────────────────────────────
def test_infer_must_decision_tree():
    assert acc.infer_must("the list must be 비어 있지 않") == "non_empty"
    assert acc.infer_must("returns at least one row") == "non_empty"
    assert acc.infer_must("the field must be 존재") == "exists"
    assert acc.infer_must("status must == 'done'") == "equals"
    assert acc.infer_must("something vague") == "non_empty"  # conservative default


# ── §2.3 prose derivation (http_read) ────────────────────────────────────────
def test_derive_prose_http_read_non_empty(tmp_path):
    root = _make_codebase(tmp_path)
    prose = "GET /api/v1/projects must return a non-empty `modules` field"
    s = acc.derive_contract_from_prose(prose, root, "AC1")
    assert s is not None and s.kind == "http_read"
    assert s.full_path == "/api/v1/projects"
    assert s.field == "modules" and s.container == "projects"
    assert s.json_path == "projects[].modules" and s.must == "non_empty"


def test_derive_prose_two_fields_resolve_multifield(tmp_path):
    # box-2 (group 0067, level-3, GAP-2): two backticked fields with a UNIFORM, literal-free
    # must is field cross-validation, no longer ambiguous — one gate asserts both fields.
    root = _make_codebase(tmp_path)
    prose = "GET /api/v1/projects must return non-empty `modules` and `tags`"
    s = acc.derive_contract_from_prose(prose, root, "AC1")
    assert s is not None and s.kind == "http_read"
    assert [a["json_path"] for a in s.asserts] == ["projects[].modules", "projects[].tags"]
    assert all(a["must"] == "non_empty" for a in s.asserts)


def test_derive_prose_multifield_with_literal_declines(tmp_path):
    # A literal + ≥2 fields can't attribute "which field == the value" → still declines
    # (box-0's never-guess trust basis, NR0003 §3).
    root = _make_codebase(tmp_path)
    prose = "GET /api/v1/projects `total` and `count` must be 정확히 3"
    assert acc.derive_contract_from_prose(prose, root, "AC1") is None


def test_derive_prose_declines_without_route(tmp_path):
    root = _make_codebase(tmp_path)
    prose = "the `modules` field must be non-empty"  # no path/verb → kind unknown
    assert acc.derive_contract_from_prose(prose, root, "AC1") is None


def test_derive_prose_equals_requires_literal(tmp_path):
    root = _make_codebase(tmp_path)
    # 'equals' modality but no extractable literal → decline
    prose = "GET /api/v1/projects `count` must be 정확히 the right number"
    assert acc.derive_contract_from_prose(prose, root, "AC1") is None


# ── §2.2 explicit oracle ─────────────────────────────────────────────────────
def test_validate_explicit_oracle_http_equals(tmp_path):
    root = _make_codebase(tmp_path)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/projects",
         "json_path": "total", "must": "equals", "expected": 3}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    assert s is not None and s.must == "equals" and s.expected == 3
    assert s.full_path == "/api/v1/projects"


def test_validate_explicit_oracle_equals_needs_expected(tmp_path):
    root = _make_codebase(tmp_path)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/projects",
         "json_path": "total", "must": "equals"}  # no expected
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_validate_explicit_oracle_unit_value(tmp_path):
    root = _make_codebase(tmp_path)
    (tmp_path / "code" / "app" / "calc.py").write_text(
        "def answer():\n    return 42\n", encoding="utf-8")
    o = {"kind": "unit_value", "target": "app/calc.py::answer",
         "must": "equals", "expected": 42}
    s = acc.validate_explicit_oracle(o, root, "AC2")
    assert s is not None and s.kind == "unit_value" and s.target == "app/calc.py::answer"


def test_validate_explicit_oracle_unit_value_missing_symbol(tmp_path):
    root = _make_codebase(tmp_path)
    o = {"kind": "unit_value", "target": "app/nope.py::ghost",
         "must": "equals", "expected": 1}
    assert acc.validate_explicit_oracle(o, root, "AC2") is None


# ── §2.4 synthesis ───────────────────────────────────────────────────────────
def test_synthesize_http_non_empty_with_discovered_fixture(tmp_path):
    root = _make_codebase(tmp_path, with_fixture=True)
    s = acc.detect_acceptance(
        {"id": "AC1",
         "prose": "GET /api/v1/projects must return a non-empty `modules` field"}, root)
    out = acc.synthesize_acceptance_red_test(s, root)
    assert out is not None
    edit = out["edit"]
    assert edit["id"] == "ACCEPTANCE_RED" and edit["kind"] == "create_file"
    assert edit["file"] == "tests/test_acceptance_modules.py"
    assert "client.get('/api/v1/projects')" in edit["content"]
    assert "def client" not in edit["content"]  # reuses the conftest fixture
    assert out["node"].startswith("tests/test_acceptance_modules.py::")


def test_synthesize_http_equals_emits_value_assert(tmp_path):
    root = _make_codebase(tmp_path, with_fixture=True)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/projects",
         "json_path": "total", "must": "equals", "expected": 3}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    out = acc.synthesize_acceptance_red_test(s, root)
    assert out is not None
    content = out["edit"]["content"]
    assert "payload['total'] == 3" in content  # the box-0 value-equality assert


def test_synthesize_unit_value_needs_no_harness(tmp_path):
    root = _make_codebase(tmp_path, with_fixture=False)  # no TestClient at all
    s = acc.AcceptanceSymptom(kind="unit_value", source_ac_id="AC2",
                              target="app/calc.py::answer", must="equals", expected=42)
    out = acc.synthesize_acceptance_red_test(s, root)
    assert out is not None
    assert "getattr(_m, 'answer') == 42" in out["edit"]["content"]
    assert out["node"].endswith("::test_acceptance_unit_answer")


def test_synthesize_http_fail_open_without_harness(tmp_path):
    root = _make_codebase(tmp_path, with_fixture=False)  # no fixture, no setup_block
    s = acc.detect_acceptance(
        {"id": "AC1",
         "prose": "GET /api/v1/projects must return a non-empty `modules` field"}, root)
    assert s is not None  # symptom resolves...
    assert acc.synthesize_acceptance_red_test(s, root) is None  # ...but no harness → decline


def test_synthesize_none_symptom_is_none(tmp_path):
    root = _make_codebase(tmp_path)
    assert acc.synthesize_acceptance_red_test(None, root) is None
