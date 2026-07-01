"""box-3 (group 0068, level-4, GAP-3): multi-step POST→GET within ONE criterion.

Level-3 (group 0067) lifted the ceiling INSIDE one fetch — N fields on ONE read certify
together. Level-4 lifts the ORTHOGONAL ceiling across the TIME axis: a criterion whose
satisfaction requires STATE CHANGE then OBSERVATION (create a resource, then read it back)
is certified by an ORDERED call sequence — POST then GET — inside ONE test = ONE node.
This is NOT a multi-gate: ``verify.py`` is untouched, the spec stays single-node (no
``red_test_nodes`` key), and a single-shot criterion is byte-identical to level-1/2/3.

Trust basis (inherited): a request BODY can NOT be derived from prose without guessing, so
multi-step is EXPLICIT-ORACLE ONLY (a ``steps:`` block the author wrote) — a prose criterion
naming two routes declines, and box-0 never fabricates a payload. The load-bearing cases:
(a) an explicit ``steps:`` oracle grounds every route and lifts the body verbatim, (b) the
synthesised test issues BOTH calls in sequence as ONE node, (c) a single-step / bodyless-guess
/ ungrounded / assertion-less oracle each declines, (d) prose multi-step declines.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import acceptance_synth as acc
from hive import specify


# ── on-disk codebase: POST + GET on the same route (mirrors test_acceptance_multifield) ──
def _codebase(tmp_path):
    """POST /api/v1/projects (create) + GET /api/v1/projects (list) + conftest TestClient."""
    root = tmp_path / "code"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1")\n'
        "_STORE = []\n\n"
        '@router.post("/projects", status_code=201)\n'
        "def create(project: dict):\n"
        '    return {"ok": True}\n\n'
        '@router.get("/projects")\n'
        "def list_projects():\n"
        '    return {"items": _STORE}\n',
        encoding="utf-8")
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


def _steps_oracle():
    return {"kind": "http_read", "steps": [
        {"verb": "post", "route": "/api/v1/projects",
         "body": {"name": "alpha"}, "expect_status": 201},
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items[].name", "must": "non_empty"}]}]}


# ── validation: explicit steps oracle grounds → a multi-step symptom ─────────
def test_explicit_steps_oracle_grounds(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_steps_oracle(), root, "AC1")
    assert s is not None and s.kind == "http_read"
    assert len(s.steps) == 2
    assert s.steps[0]["verb"] == "post" and s.steps[0]["body"] == {"name": "alpha"}
    assert s.steps[0]["expect_status"] == 201
    # terminal read carries the assertions; the scalar slots mirror its first assert.
    assert s.steps[1]["verb"] == "get"
    assert s.steps[1]["asserts"][0]["json_path"] == "items[].name"
    assert s.json_path == "items[].name" and s.asserts == ()


def test_steps_verb_is_the_authors_not_the_resolved(tmp_path):
    # the SAME path resolves both a POST and a GET binding; box-0 must honour the AUTHOR's
    # per-step verb, never collapse to a single resolved verb.
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_steps_oracle(), root, "AC1")
    assert [st["verb"] for st in s.steps] == ["post", "get"]


# ── decline gates (never-guess / fail-open) ──────────────────────────────────
def test_single_step_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "steps": [
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items[].name", "must": "non_empty"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None  # <2 steps → not multi-step


def test_terminal_without_assertions_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "steps": [
        {"verb": "post", "route": "/api/v1/projects", "body": {"name": "x"}},
        {"verb": "get", "route": "/api/v1/projects"}]}  # no asserts / json_path
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_non_dict_body_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "steps": [
        {"verb": "post", "route": "/api/v1/projects", "body": "alpha"},  # scalar body = guess
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items[].name", "must": "non_empty"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_ungrounded_step_route_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "steps": [
        {"verb": "post", "route": "/api/v1/nonexistent", "body": {"name": "x"}},
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items[].name", "must": "non_empty"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_bad_terminal_assert_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "steps": [
        {"verb": "post", "route": "/api/v1/projects", "body": {"name": "x"}},
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items[].name", "must": "equals"}]}]}  # equals w/o expected
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_prose_never_yields_multistep(tmp_path):
    # prose can't spell a request body, so it NEVER produces a multi-step symptom — at most
    # a harmless single-shot read. The steps (time) axis is explicit-oracle only (never-guess).
    root = _codebase(tmp_path)
    prose = "POST /api/v1/projects then GET /api/v1/projects must return `items`"
    s = acc.derive_contract_from_prose(prose, root, "AC1")
    assert s is None or s.steps == ()


# ── synthesis: multi-step is ONE test body, ONE node, ONE file ───────────────
def test_multistep_body_has_ordered_calls(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_steps_oracle(), root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    assert res is not None
    content = res["edit"]["content"]
    # ordered POST→GET, body lifted VERBATIM, single def, one node.
    assert content.count(".post(") == 1 and content.count(".get(") == 1
    assert "json={'name': 'alpha'}" in content        # never guessed — author's body
    assert "assert r1.status_code == 201" in content  # expect_status honoured
    assert "'items' in payload" in content            # terminal read assertion
    assert content.count("def test_acceptance_multistep") == 1
    assert "::test_acceptance_multistep" in res["node"]


def test_multistep_equals_appends_value_assert(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "steps": [
        {"verb": "post", "route": "/api/v1/projects", "body": {"name": "x"}},
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "total", "must": "equals", "expected": 1}]}]}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    assert res is not None
    assert "== 1, 'acceptance value mismatch'" in res["edit"]["content"]


def test_default_status_check_without_expect_status(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "steps": [
        {"verb": "post", "route": "/api/v1/projects", "body": {"name": "x"}},  # no expect_status
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items[].name", "must": "non_empty"}]}]}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    assert "assert 200 <= r1.status_code < 300" in res["edit"]["content"]


# ── specify wiring: a multi-step criterion is a SINGLE gate ──────────────────
def test_specify_multistep_is_single_gate(tmp_path):
    root = _codebase(tmp_path)
    design = (
        "# Feature\n\n## 수용기준\n"
        "- id: AC1\n"
        "  prose: creating a project makes it appear in the list\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    steps:\n"
        "      - verb: post\n"
        "        route: /api/v1/projects\n"
        "        body: {name: \"alpha\"}\n"
        "        expect_status: 201\n"
        "      - verb: get\n"
        "        route: /api/v1/projects\n"
        "        asserts:\n"
        "          - json_path: items[].name\n"
        "            must: non_empty\n")
    spec = {"edits": [{"id": "E1", "file": "app/routes.py",
                       "anchor_old": '{"items": _STORE}', "replacement_new": '{"items": _STORE}'}]}
    spec = specify._synthesize_acceptance_red_test(spec, design, root, app_fixture="client")
    v = spec["verify"]
    assert v.get("red_test_node")            # a gate was wired
    assert "red_test_nodes" not in v         # ONE gate, not a multi-gate (level-4 ≠ level-2)
    accept = [e for e in spec["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    assert len(accept) == 1                  # ONE test file
    assert accept[0]["content"].count(".post(") == 1
    assert accept[0]["content"].count(".get(") == 1


# ── ablation: a single-shot criterion is untouched (byte-shape preserved) ────
def test_single_shot_criterion_carries_no_steps(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/projects",
         "json_path": "items", "must": "non_empty"}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    assert s is not None and s.steps == () and s.asserts == ()  # level-1 shape intact
