"""box-5b (group 0072, level-6b, GAP-5b): a relational invariant ACROSS independent routes.

Level-6 (group 0070) lifted the ceiling across the RELATION axis — one observed value agrees
with ANOTHER observed value on the SAME payload (``total == len(items)``). Level-5 (group 0069)
lifted it across the ROUTE axis — a conjunction over ≥2 INDEPENDENT read routes. box-5b is their
COMPOSITION: a relation whose two sides live on DIFFERENT read payloads (a summary route's count
equals the length of the list route it summarises). This is NOT a new axis and NOT a new must —
the same ``equals_len``/``equals_path``, but the RHS carries an ``other_read`` INDEX selecting
which read supplies ``other_path``. It stays ONE test = ONE node: ``verify.py``/``specify.py`` are
untouched, the spec is single-node, and a non-cross criterion is byte-identical to level-1..6.

Trust basis (inherited): which read's which field relates to which can NOT be resolved from prose
without guessing, so cross-route is EXPLICIT-ORACLE ONLY. The load-bearing cases: (a) an explicit
cross-route ``reads`` oracle grounds every route and lifts the relation verbatim, (b) the
synthesised test fetches EVERY payload FIRST then asserts across them (a cross-route assert can not
reference a payload not yet fetched — box-4's inline-assert builder can not express this, hence a
dedicated two-pass builder), (c) the self-reference guard is RELAXED for cross-read (the same field
name on a DIFFERENT route is legitimate) but an out-of-range / self index declines, (d) an
``other_read`` outside a reads context (single route, steps terminal read) declines, (e) a reads
oracle WITHOUT ``other_read`` still uses the box-4 builder (byte-identical), (f) specify wires it as
a SINGLE gate.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import acceptance_synth as acc
from hive import specify


# ── on-disk codebase: a summary route (count) + a list route it summarises ───
def _codebase(tmp_path):
    """GET /api/v1/dashboard/summary -> {project_count}; GET /api/v1/projects -> {items}."""
    root = tmp_path / "code"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1")\n'
        "_STORE = [{'id': 1}]\n\n"
        '@router.get("/projects")\n'
        "def list_projects():\n"
        '    return {"items": _STORE}\n\n'
        '@router.get("/dashboard/summary")\n'
        "def summary():\n"
        '    return {"project_count": len(_STORE)}\n',
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


def _cross_oracle():
    """summary.project_count == len(projects.items): read#0 relates to read#1 (other_read: 1)."""
    return {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "equals_len",
                      "other_path": "items", "other_read": 1}]},
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]}]}


# ── validation: an explicit cross-route relation oracle grounds ──────────────
def test_cross_route_oracle_grounds(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_cross_oracle(), root, "AC1")
    assert s is not None and s.kind == "http_read"
    assert len(s.reads) == 2
    a0 = s.reads[0]["asserts"][0]
    assert a0["must"] == "equals_len" and a0["other_path"] == "items"
    assert a0["other_read"] == 1                       # the cross-read locator survived
    assert s.reads[0]["full_path"] == "/api/v1/dashboard/summary"
    assert s.reads[1]["full_path"] == "/api/v1/projects"


# ── synthesis: fetch EVERY payload first, THEN assert across them ────────────
def test_cross_route_body_is_two_pass_and_relates_across_payloads(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_cross_oracle(), root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    assert res is not None
    content = res["edit"]["content"]
    # both payloads fetched, the relation compares payload1's count to len(payload2's items).
    assert "payload1 = r1.json()" in content and "payload2 = r2.json()" in content
    assert "payload1['project_count'] == len(payload2['items'])" in content
    # two-pass ordering (load-bearing): the cross-payload assert must appear AFTER BOTH
    # payloads are assigned — box-4's inline builder would emit it before payload2 exists.
    rel = content.index("payload1['project_count'] == len(payload2['items'])")
    assert content.index("payload2 = r2.json()") < rel
    # the RHS list-guard is on the OTHER payload (payload2), not the LHS payload.
    assert "isinstance(payload2['items'], list)" in content
    # distinct fn stem + ONE node.
    assert content.count("def test_acceptance_multiroute_relation") == 1
    assert "::test_acceptance_multiroute_relation" in res["node"]


def test_cross_route_equals_path_relates_two_scalars(tmp_path):
    """equals_path across routes: a value on one route equals a value on another."""
    root = tmp_path / "code"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1")\n'
        '@router.get("/a")\n'
        "def a():\n"
        '    return {"total": 3}\n\n'
        '@router.get("/b")\n'
        "def b():\n"
        '    return {"total": 3}\n',
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
    # SAME field name `total` on DIFFERENT routes — the self-reference guard must be RELAXED.
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/a",
         "asserts": [{"json_path": "total", "must": "equals_path",
                      "other_path": "total", "other_read": 1}]},
        {"verb": "get", "route": "/api/v1/b",
         "asserts": [{"json_path": "total", "must": "exists"}]}]}
    s = acc.validate_explicit_oracle(o, str(root), "AC1")
    assert s is not None                               # same-name cross-route is legitimate
    res = acc.synthesize_acceptance_red_test(s, str(root), app_fixture="client", test_dir="tests")
    content = res["edit"]["content"]
    assert "payload1['total'] == payload2['total']" in content


# ── decline gates (never-guess / fail-open) ──────────────────────────────────
def test_other_read_out_of_range_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "equals_len",
                      "other_path": "items", "other_read": 5}]},   # only 2 reads → 5 is oob
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_other_read_self_index_declines(tmp_path):
    # a relation of a read with ITSELF is a within-payload box-5 relation (omit other_read).
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "equals_len",
                      "other_path": "items", "other_read": 0}]},    # points at itself
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_other_read_bool_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "equals_len",
                      "other_path": "items", "other_read": True}]},  # bool is not an index
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_other_read_negative_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "equals_len",
                      "other_path": "items", "other_read": -1}]},
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_other_read_on_single_route_declines(tmp_path):
    # a cross-read locator on a single-route (box-2) oracle is meaningless — one payload only.
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "equals_len",
                      "other_path": "items", "other_read": 1}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_other_read_in_steps_terminal_read_declines(tmp_path):
    # a cross-read locator inside a level-4 ``steps`` terminal read has no second payload.
    root = tmp_path / "code"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1")\n'
        "_S = []\n\n"
        '@router.post("/projects", status_code=201)\n'
        "def create(p: dict):\n"
        '    return {"ok": True}\n\n'
        '@router.get("/projects")\n'
        "def lst():\n"
        '    return {"items": _S, "count": len(_S)}\n',
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
    o = {"kind": "http_read", "steps": [
        {"verb": "post", "route": "/api/v1/projects", "body": {"name": "x"}},
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "count", "must": "equals_len",
                      "other_path": "items", "other_read": 0}]}]}
    assert acc.validate_explicit_oracle(o, str(root), "AC1") is None


def test_cross_read_iterating_path_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "items[].id", "must": "equals_len",
                      "other_path": "items", "other_read": 1}]},   # per-item path in a relation
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_cross_read_with_expected_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "equals_len",
                      "other_path": "items", "other_read": 1, "expected": 3}]},  # value+relation
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_prose_never_yields_cross_route(tmp_path):
    root = _codebase(tmp_path)
    prose = ("the summary `project_count` on GET /api/v1/dashboard/summary must equal the "
             "number of `items` on GET /api/v1/projects")
    s = acc.derive_contract_from_prose(prose, root, "AC1")
    assert s is None or (s.reads == () and s.asserts == ())


# ── ablation: a reads oracle WITHOUT other_read stays box-4 (byte-identical) ──
def test_reads_without_other_read_uses_box4_builder(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "exists"}]},
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]}]}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    content = res["edit"]["content"]
    # no cross-read → the plain box-4 stem, no cross-payload relation, inline-assert shape.
    assert "def test_acceptance_multiroute_" in content
    assert "def test_acceptance_multiroute_relation" not in content
    assert "len(payload2[" not in content


# ── specify wiring: a cross-route criterion is a SINGLE gate ──────────────────
def test_specify_cross_route_is_single_gate(tmp_path):
    root = _codebase(tmp_path)
    design = (
        "# Feature\n\n## 수용기준\n"
        "- id: AC1\n"
        "  prose: the summary count must equal the number of listed projects\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    reads:\n"
        "      - verb: get\n"
        "        route: /api/v1/dashboard/summary\n"
        "        asserts:\n"
        "          - json_path: project_count\n"
        "            must: equals_len\n"
        "            other_path: items\n"
        "            other_read: 1\n"
        "      - verb: get\n"
        "        route: /api/v1/projects\n"
        "        asserts:\n"
        "          - json_path: items\n"
        "            must: non_empty\n")
    spec = {"edits": [{"id": "E1", "file": "app/routes.py",
                       "anchor_old": '{"items": _STORE}', "replacement_new": '{"items": _STORE}'}]}
    spec = specify._synthesize_acceptance_red_test(spec, design, root, app_fixture="client")
    v = spec["verify"]
    assert v.get("red_test_node")            # a gate was wired
    assert "red_test_nodes" not in v         # ONE gate, not a multi-gate
    accept = [e for e in spec["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    assert len(accept) == 1
    assert "len(payload2['items'])" in accept[0]["content"]
