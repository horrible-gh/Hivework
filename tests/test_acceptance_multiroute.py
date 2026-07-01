"""box-4 (group 0069, level-5, GAP-4): multiple INDEPENDENT routes within ONE criterion.

Level-4 (group 0068) lifted the ceiling across the TIME axis — an ORDERED POST→GET causal
sequence certifies one criterion. Level-5 lifts the ORTHOGONAL ceiling across the ROUTE
axis: a criterion whose satisfaction is the CONJUNCTION of observations on ≥2 UNORDERED,
INDEPENDENT read routes (the item appears in ``GET /projects`` AND the count is reflected in
``GET /dashboard/summary``) is certified by issuing every read and asserting each — inside
ONE test = ONE node. This is NOT a multi-gate: ``verify.py`` is untouched, the spec stays
single-node (no ``red_test_nodes`` key), and a single-route criterion is byte-identical to
level-1/2/3/4.

Trust basis (inherited): which N INDEPENDENT routes a prose criterion means can NOT be
resolved without guessing, so multi-route is EXPLICIT-ORACLE ONLY (a ``reads:`` block the
author wrote) — a prose criterion naming two routes declines. A ``reads`` entry must be a
READ verb (state change belongs in level-4 ``steps``, not an unordered observation set). The
load-bearing cases: (a) an explicit ``reads:`` oracle grounds every route and lifts every
assert verbatim, (b) the synthesised test fetches each route into its OWN payload as ONE
node, (c) a single-read / mutating-verb / ungrounded / assertion-less oracle each declines,
(d) prose multi-route declines.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import acceptance_synth as acc
from hive import specify


# ── on-disk codebase: TWO independent read routes + conftest TestClient ──────
def _codebase(tmp_path):
    """GET /api/v1/projects (list) + GET /api/v1/dashboard/summary (count) + POST (create)."""
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


def _reads_oracle():
    return {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]},
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "exists"}]}]}


# ── validation: explicit reads oracle grounds → a multi-route symptom ────────
def test_explicit_reads_oracle_grounds(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_reads_oracle(), root, "AC1")
    assert s is not None and s.kind == "http_read"
    assert len(s.reads) == 2
    assert s.reads[0]["verb"] == "get" and s.reads[0]["full_path"] == "/api/v1/projects"
    assert s.reads[0]["asserts"][0]["json_path"] == "items"
    assert s.reads[1]["full_path"] == "/api/v1/dashboard/summary"
    assert s.reads[1]["asserts"][0]["json_path"] == "project_count"
    # the scalar slots mirror the FIRST read's first assert; reads is the sole signal.
    assert s.json_path == "items" and s.steps == () and s.asserts == ()


def test_reads_routes_are_independent_not_ordered(tmp_path):
    # both reads resolve distinct full_paths; there is no terminal/mutation distinction
    # (unlike level-4 steps) — every read carries its own asserts.
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_reads_oracle(), root, "AC1")
    assert [r["full_path"] for r in s.reads] == [
        "/api/v1/projects", "/api/v1/dashboard/summary"]
    assert all(r["asserts"] for r in s.reads)


# ── decline gates (never-guess / fail-open) ──────────────────────────────────
def test_single_read_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None  # <2 reads → not multi-route


def test_mutating_verb_in_reads_declines(tmp_path):
    # a POST inside reads would change state — that belongs in a level-4 ``steps`` sequence,
    # not an unordered observation set. The route axis is read-only.
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "post", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]},
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "exists"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_ungrounded_read_route_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]},
        {"verb": "get", "route": "/api/v1/nonexistent",
         "asserts": [{"json_path": "x", "must": "exists"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_read_without_assertions_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/projects"},  # no asserts / json_path
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "exists"}]}]}
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_reads_equals_without_expected_declines(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]},
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "equals"}]}]}  # equals w/o expected
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_prose_never_yields_multiroute(tmp_path):
    # prose can't pin which N independent routes, so it NEVER produces a multi-route symptom —
    # at most a harmless single-shot read. The route axis is explicit-oracle only (never-guess).
    root = _codebase(tmp_path)
    prose = ("GET /api/v1/projects returns `items` and "
             "GET /api/v1/dashboard/summary returns `project_count`")
    s = acc.derive_contract_from_prose(prose, root, "AC1")
    assert s is None or s.reads == ()


# ── synthesis: multi-route is ONE test body, ONE node, ONE file ──────────────
def test_multiroute_body_fetches_each_route_into_own_payload(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_reads_oracle(), root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    assert res is not None
    content = res["edit"]["content"]
    # two independent GET calls, each into its OWN payload, single def, one node.
    assert content.count(".get(") == 2 and content.count(".post(") == 0
    assert "r1 = client.get('/api/v1/projects')" in content
    assert "r2 = client.get('/api/v1/dashboard/summary')" in content
    assert "payload1 = r1.json()" in content and "payload2 = r2.json()" in content
    assert "'items' in payload1" in content
    assert "'project_count' in payload2" in content
    assert content.count("def test_acceptance_multiroute") == 1
    assert "::test_acceptance_multiroute" in res["node"]


def test_multiroute_equals_appends_value_assert_on_right_payload(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "reads": [
        {"verb": "get", "route": "/api/v1/projects",
         "asserts": [{"json_path": "items", "must": "non_empty"}]},
        {"verb": "get", "route": "/api/v1/dashboard/summary",
         "asserts": [{"json_path": "project_count", "must": "equals", "expected": 1}]}]}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    assert res is not None
    content = res["edit"]["content"]
    # the equals compare must target payload2 (the summary route), not payload1.
    assert "payload2['project_count'] == 1, 'acceptance value mismatch'" in content


# ── specify wiring: a multi-route criterion is a SINGLE gate ─────────────────
def test_specify_multiroute_is_single_gate(tmp_path):
    root = _codebase(tmp_path)
    design = (
        "# Feature\n\n## 수용기준\n"
        "- id: AC1\n"
        "  prose: creating a project reflects in both the list and the summary count\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    reads:\n"
        "      - verb: get\n"
        "        route: /api/v1/projects\n"
        "        asserts:\n"
        "          - json_path: items\n"
        "            must: non_empty\n"
        "      - verb: get\n"
        "        route: /api/v1/dashboard/summary\n"
        "        asserts:\n"
        "          - json_path: project_count\n"
        "            must: exists\n")
    spec = {"edits": [{"id": "E1", "file": "app/routes.py",
                       "anchor_old": '{"items": _STORE}', "replacement_new": '{"items": _STORE}'}]}
    spec = specify._synthesize_acceptance_red_test(spec, design, root, app_fixture="client")
    v = spec["verify"]
    assert v.get("red_test_node")            # a gate was wired
    assert "red_test_nodes" not in v         # ONE gate, not a multi-gate (level-5 ≠ level-2)
    accept = [e for e in spec["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    assert len(accept) == 1                  # ONE test file
    assert accept[0]["content"].count(".get(") == 2


# ── ablation: a single-route criterion is untouched (byte-shape preserved) ───
def test_single_route_criterion_carries_no_reads(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/projects",
         "json_path": "items", "must": "non_empty"}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    assert s is not None and s.reads == () and s.steps == () and s.asserts == ()  # level-1 shape
