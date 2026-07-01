"""box-5 (group 0070, level-6, GAP-5): the RELATIONSHIP axis within ONE criterion.

Levels 2-5 all certify an observation against a CONSTANT truth: a field exists, is
non-empty, or ``== <literal>``. Level-6 lifts the ORTHOGONAL ceiling across the RELATION
axis: a criterion satisfied only when one observed value agrees with ANOTHER observed value
— the cross-field/cross-entity invariant NR0003 named (``total == len(items)``: a count
field must equal the length of the list it summarises). This is NOT a new gate: ``verify.py``
and ``specify.py`` are untouched, the spec stays single-node (no ``red_test_nodes``), and a
non-relational criterion is byte-identical to level-1/2/3/4/5.

Trust basis (inherited): which two paths a prose criterion relates can NOT be resolved
without guessing, so a relation is EXPLICIT-ORACLE ONLY (an ``equals_len`` / ``equals_path``
assert the author wrote) — a prose criterion declines. The RHS is a companion ``other_path``,
never a literal (a value + a relation is ambiguous), never an iterating ``[]`` path (a
relation is over whole values / list lengths, not per item). The load-bearing cases: (a) an
explicit relational oracle grounds and lifts verbatim, (b) the synthesised test emits
``left == len(other)`` with an ``isinstance(list)`` guard, (c) missing/self/literal-clashing/
iterating oracles each decline, (d) prose never yields a relation, (e) a relation rides inside
level-4 ``steps`` and level-5 ``reads`` unchanged, (f) specify wires it as a SINGLE gate.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import acceptance_synth as acc
from hive import specify


# ── on-disk codebase: a summary route returning BOTH a count and its list ────
def _codebase(tmp_path):
    """GET /api/v1/dashboard/summary -> {total, items}; GET /api/v1/projects; POST create."""
    root = tmp_path / "code"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1")\n'
        "_STORE = [{'id': 1}]\n\n"
        '@router.post("/projects", status_code=201)\n'
        "def create(project: dict):\n"
        '    return {"ok": True}\n\n'
        '@router.get("/projects")\n'
        "def list_projects():\n"
        '    return {"items": _STORE}\n\n'
        '@router.get("/dashboard/summary")\n'
        "def summary():\n"
        '    return {"total": len(_STORE), "items": _STORE, "label": "ok", "count2": 1}\n',
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


def _rel_oracle(must="equals_len", other="items", **extra):
    a = {"json_path": "total", "must": must, "other_path": other}
    a.update(extra)
    return {"kind": "http_read", "verb": "get",
            "route": "/api/v1/dashboard/summary", "asserts": [a]}


# ── validation: an explicit relational oracle grounds → a relation symptom ────
def test_relational_oracle_grounds(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_rel_oracle(), root, "AC1")
    assert s is not None
    assert s.full_path == "/api/v1/dashboard/summary"
    # a SINGLE relational assert does NOT reduce to the scalar shape (no other_path slot).
    assert len(s.asserts) == 1
    assert s.asserts[0] == {"json_path": "total", "must": "equals_len",
                            "other_path": "items", "expected": None}
    # scalar slots mirror the first assert for the fn name / rationale.
    assert s.json_path == "total" and s.must == "equals_len"


def test_normalize_one_assert_lifts_relation():
    a = acc._normalize_one_assert(
        {"json_path": "total", "must": "equals_len", "other_path": "items"})
    assert a == {"json_path": "total", "must": "equals_len",
                 "other_path": "items", "expected": None}


# ── synthesis: equals_len emits `left == len(other)` with an isinstance guard ─
def test_equals_len_builds_relation_assert(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_rel_oracle(), root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    assert res is not None
    content = res["edit"]["content"]
    # one fetch, existence guards on BOTH paths, list guard on the RHS, then the relation.
    assert content.count(".get(") == 1 and content.count(".post(") == 0
    assert "payload = response.json()" in content
    assert "'total' in payload" in content and "'items' in payload" in content
    assert "assert isinstance(payload['items'], list)" in content
    assert "assert payload['total'] == len(payload['items'])" in content
    # single test = single node; the file/fn is tagged as a relation gate.
    assert content.count("def test_acceptance_multifield") == 1
    assert "_relation" in res["edit"]["file"]
    assert "box-5 level-6 relation" in res["edit"]["rationale"]


def test_equals_path_builds_value_equality(tmp_path):
    root = _codebase(tmp_path)
    o = _rel_oracle(must="equals_path", other="count2")
    s = acc.validate_explicit_oracle(o, root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    assert res is not None
    content = res["edit"]["content"]
    assert "assert payload['total'] == payload['count2']" in content
    assert "len(" not in content            # equals_path is a plain value compare


# ── declines: the never-guess boundary ───────────────────────────────────────
def test_relation_without_other_path_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "total", "must": "equals_len"}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_relation_self_reference_declines(tmp_path):
    root = _codebase(tmp_path)
    o = _rel_oracle(other="total")          # other_path == json_path → trivial, decline
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_relation_with_literal_expected_declines(tmp_path):
    root = _codebase(tmp_path)
    o = _rel_oracle(expected=3)             # a value AND a relation → ambiguous, decline
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_relation_iterating_path_declines(tmp_path):
    root = _codebase(tmp_path)
    o = _rel_oracle(other="items[]")        # a relation is over the whole list, not per item
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_prose_never_yields_relation(tmp_path):
    root = _codebase(tmp_path)
    prose = "the `total` must equal the number of `items` in the summary"
    s = acc.derive_contract_from_prose(prose, root, "AC1")
    # prose can name two backticked fields but can NOT be lifted to a relation (never-guess):
    # it resolves to at most a non-relational shape, never an equals_len/equals_path.
    assert s is None or all(
        a.get("must") not in acc.SUPPORTED_MUST_BOX5_REL for a in (s.asserts or ()))
    assert s is None or s.must not in acc.SUPPORTED_MUST_BOX5_REL


# ── the relation rides the level-4 (steps) and level-5 (reads) shapes ────────
def test_relation_inside_reads_lifts_and_builds(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]},
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "total", "must": "equals_len", "other_path": "items"}]}]}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    assert s is not None and len(s.reads) == 2
    assert s.reads[1]["asserts"][0]["other_path"] == "items"
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    content = res["edit"]["content"]
    # the relation compares within the SECOND read's own payload.
    assert "assert payload2['total'] == len(payload2['items'])" in content
    assert "isinstance(payload2['items'], list)" in content


def test_relation_inside_steps_terminal_read_lifts(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "steps": [
        {"verb": "post", "route": "/api/v1/projects", "body": {"name": "x"},
         "expect_status": 201},
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "total", "must": "equals_len", "other_path": "items"}]}]}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    assert s is not None and len(s.steps) == 2
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    content = res["edit"]["content"]
    assert content.count(".post(") == 1 and content.count(".get(") == 1
    assert "assert payload['total'] == len(payload['items'])" in content


# ── specify wiring: a relational criterion is a SINGLE gate ──────────────────
def test_specify_relation_is_single_gate(tmp_path):
    root = _codebase(tmp_path)
    design = (
        "# Feature\n\n## 수용기준\n"
        "- id: AC1\n"
        "  prose: the summary total must equal the number of items\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    verb: get\n"
        "    route: /api/v1/dashboard/summary\n"
        "    asserts:\n"
        "      - json_path: total\n"
        "        must: equals_len\n"
        "        other_path: items\n")
    spec = {"edits": [{"id": "SRC", "kind": "edit", "file": "app/routes.py",
                       "rationale": "feature"}],
            "verify": {}}
    out = specify._synthesize_acceptance_red_test(spec, design, root, app_fixture="client")
    v = out["verify"]
    assert v.get("red_test_node")               # a single gate is wired
    assert not v.get("red_test_nodes")          # NOT a multi-gate (verify.py untouched)
    files = [e["file"] for e in out["edits"] if e.get("kind") == "create_file"]
    assert len(files) == 1 and "_relation" in files[0]


# ── byte-identity: a NON-relational single assert still reduces to scalar ─────
def test_single_nonrelational_assert_still_reduces_to_scalar(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "total", "must": "exists"}]}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    assert s is not None
    assert s.asserts == ()                      # reduced — level-1/2 output is byte-identical
    assert s.json_path == "total" and s.must == "exists"
