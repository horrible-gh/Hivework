"""box-6 (group 0071, level-7, GAP-6): the DELTA axis within ONE criterion.

Levels 2-6 all certify an observation at a SINGLE point in time: a field exists, is
non-empty, ``== <literal>``, or equals ANOTHER value on the same payload. Level-7 lifts the
ORTHOGONAL ceiling across the DELTA axis: a criterion satisfied only when a mutation moves an
observable by a specific amount (creating a project raises the dashboard count by exactly one).
This is distinct from level-4 (``steps``): box-3 asserts the ABSOLUTE post-state, box-6 asserts
the MAGNITUDE OF CHANGE. It is NOT a new gate: ``verify.py`` and ``specify.py`` are untouched,
the spec stays single-node (no ``red_test_nodes``), and a non-delta criterion is byte-identical
to level-1..6.

Trust basis (inherited): which scalar a mutation should move, and by how much, can NOT be
resolved from prose without guessing, so a delta is EXPLICIT-ORACLE ONLY (a ``delta:`` block the
author wrote). The observe half must be a read verb; the mutate half a mutating verb; ``by`` a
number (never a bool); the json_path a scalar (never an iterating ``[]``). The load-bearing
cases: (a) an explicit delta oracle grounds both routes and lifts verbatim, (b) the synthesised
test reads BEFORE, mutates, reads AFTER, and asserts the signed difference, (c) the three musts
emit ``+by`` / ``-by`` / ``after - before == by``, (d) verb-role / numeric / grounding / prose
violations each decline, (e) specify wires it as a SINGLE gate, (f) a non-delta oracle is
unaffected.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import acceptance_synth as acc
from hive import specify


# ── on-disk codebase: a summary count + a create/list pair on /projects ──────
def _codebase(tmp_path):
    """GET /api/v1/dashboard/summary -> {total}; POST + GET /api/v1/projects (the mutate/read
    pair — the mutate route grounds through the GET binding at the same path, box-3 precedent)."""
    root = tmp_path / "code"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1")\n'
        "_STORE = [{'id': 1}]\n\n"
        '@router.post("/projects", status_code=201)\n'
        "def create(project: dict):\n"
        "    _STORE.append(project)\n"
        '    return {"ok": True}\n\n'
        '@router.get("/projects")\n'
        "def list_projects():\n"
        '    return {"items": _STORE}\n\n'
        '@router.get("/dashboard/summary")\n'
        "def summary():\n"
        '    return {"total": len(_STORE)}\n',
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


def _delta_oracle(must="increases_by", by=1, **overrides):
    observe = {"verb": "get", "route": "/api/v1/dashboard/summary", "json_path": "total"}
    mutate = {"verb": "post", "route": "/api/v1/projects",
              "body": {"name": "alpha"}, "expect_status": 201}
    observe.update(overrides.pop("observe", {}))
    mutate.update(overrides.pop("mutate", {}))
    d = {"observe": observe, "mutate": mutate, "must": must, "by": by}
    d.update(overrides)
    return {"kind": "http_read", "delta": d}


# ── validation: an explicit delta oracle grounds → a delta symptom ───────────
def test_delta_oracle_grounds(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_delta_oracle(), root, "AC1")
    assert s is not None and s.kind == "http_read"
    assert len(s.delta) == 1
    spec = s.delta[0]
    assert spec["must"] == "increases_by" and spec["by"] == 1
    assert spec["observe"] == {"verb": "get",
                               "full_path": "/api/v1/dashboard/summary", "json_path": "total"}
    assert spec["mutate"]["verb"] == "post"
    assert spec["mutate"]["full_path"] == "/api/v1/projects"
    assert spec["mutate"]["body"] == {"name": "alpha"}
    assert spec["mutate"]["expect_status"] == 201
    # scalar slots mirror the observe read for the fn name / rationale.
    assert s.full_path == "/api/v1/dashboard/summary" and s.json_path == "total"
    # a delta is NOT a multi-field / multi-step / multi-route shape.
    assert s.asserts == () and s.steps == () and s.reads == ()


# ── synthesis: the three musts emit before/mutate/after with a signed delta ──
def test_increases_by_builds_before_after_delta(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_delta_oracle(by=1), root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    assert res is not None
    content = res["edit"]["content"]
    # the SAME scalar route is read twice (before and after); the mutation happens once.
    assert content.count(".get(") == 2 and content.count(".post(") == 1
    assert "before = r0.json()" in content and "after = r2.json()" in content
    assert "assert r1.status_code == 201, r1.text" in content
    assert "assert after['total'] == before['total'] + 1, " in content
    # single test = single node; the file/fn is tagged as a delta gate.
    assert content.count("def test_acceptance_delta_") == 1
    assert "_delta" in res["edit"]["file"]
    assert "box-6 level-7" in res["edit"]["rationale"]


def test_decreases_by_builds_subtraction(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_delta_oracle(must="decreases_by", by=2), root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    content = res["edit"]["content"]
    assert "assert after['total'] == before['total'] - 2, " in content


def test_delta_equals_builds_signed_difference(tmp_path):
    root = _codebase(tmp_path)
    s = acc.validate_explicit_oracle(_delta_oracle(must="delta_equals", by=-1), root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    content = res["edit"]["content"]
    assert "assert after['total'] - before['total'] == -1, " in content


def test_delta_without_body_omits_json_kwarg(tmp_path):
    root = _codebase(tmp_path)
    o = _delta_oracle(mutate={"body": None, "expect_status": None})
    s = acc.validate_explicit_oracle(o, root, "AC1")
    res = acc.synthesize_acceptance_red_test(s, root, app_fixture="client", test_dir="tests")
    content = res["edit"]["content"]
    assert "r1 = client.post('/api/v1/projects')" in content   # no json= kwarg
    assert "assert 200 <= r1.status_code < 300, r1.text" in content  # default status band


# ── declines: the never-guess boundary ───────────────────────────────────────
def test_observe_with_mutating_verb_declines(tmp_path):
    root = _codebase(tmp_path)
    # a POST can not be the OBSERVE half — a read must not change state.
    o = _delta_oracle(observe={"verb": "post"})
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_mutate_with_read_verb_declines(tmp_path):
    root = _codebase(tmp_path)
    # a GET can not be the MUTATE half — a read does not move the counter.
    o = _delta_oracle(mutate={"verb": "get"})
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_missing_by_declines(tmp_path):
    root = _codebase(tmp_path)
    o = _delta_oracle()
    del o["delta"]["by"]
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_non_numeric_by_declines(tmp_path):
    root = _codebase(tmp_path)
    assert acc.validate_explicit_oracle(_delta_oracle(by="one"), root, "AC1") is None


def test_bool_by_declines(tmp_path):
    root = _codebase(tmp_path)
    # ``by: true`` is an int subclass (True == 1) — reject it so a delta never silently means +1.
    assert acc.validate_explicit_oracle(_delta_oracle(by=True), root, "AC1") is None


def test_unsupported_must_declines(tmp_path):
    root = _codebase(tmp_path)
    assert acc.validate_explicit_oracle(_delta_oracle(must="equals"), root, "AC1") is None


def test_iterating_json_path_declines(tmp_path):
    root = _codebase(tmp_path)
    # a delta is over a scalar count, never an iterating path.
    o = _delta_oracle(observe={"json_path": "items[].id"})
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_ungrounded_mutate_route_declines(tmp_path):
    root = _codebase(tmp_path)
    o = _delta_oracle(mutate={"route": "/api/v1/nonexistent"})
    assert acc.validate_explicit_oracle(o, root, "AC1") is None


def test_prose_never_yields_delta(tmp_path):
    root = _codebase(tmp_path)
    prose = "creating a project raises the `total` by exactly 1 on the summary"
    s = acc.derive_contract_from_prose(prose, root, "AC1")
    # prose can name a field and a number but can NOT be lifted to a before/after delta
    # (never-guess): it resolves to at most a non-delta shape, never a delta symptom.
    assert s is None or not s.delta


# ── byte-identity: a non-delta oracle is unaffected ──────────────────────────
def test_non_delta_oracle_unaffected(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/dashboard/summary",
         "json_path": "total", "must": "exists"}
    s = acc.validate_explicit_oracle(o, root, "AC1")
    assert s is not None and s.delta == ()   # scalar shape, level-1/2 byte-identical
    assert s.json_path == "total" and s.must == "exists"


# ── specify wiring: a delta criterion is a SINGLE gate ───────────────────────
def test_specify_delta_is_single_gate(tmp_path):
    root = _codebase(tmp_path)
    design = (
        "# Feature\n\n## 수용기준\n"
        "- id: AC1\n"
        "  prose: creating a project raises the summary total by exactly one\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    delta:\n"
        "      observe:\n"
        "        verb: get\n"
        "        route: /api/v1/dashboard/summary\n"
        "        json_path: total\n"
        "      mutate:\n"
        "        verb: post\n"
        "        route: /api/v1/projects\n"
        "        body: {name: \"alpha\"}\n"
        "        expect_status: 201\n"
        "      must: increases_by\n"
        "      by: 1\n")
    spec = {"edits": [{"id": "SRC", "kind": "edit", "file": "app/routes.py",
                       "rationale": "feature"}],
            "verify": {}}
    out = specify._synthesize_acceptance_red_test(spec, design, root, app_fixture="client")
    v = out["verify"]
    assert v.get("red_test_node")               # a single gate is wired
    assert not v.get("red_test_nodes")          # NOT a multi-gate (verify.py untouched)
    files = [e["file"] for e in out["edits"] if e.get("kind") == "create_file"]
    assert len(files) == 1 and "_delta" in files[0]
