"""box-2 (group 0067, level-3, GAP-2): field cross-validation within ONE criterion.

Level-2 (group 0066) lifted the GATE ceiling — N acceptance criteria → N red→green gates
(N test files, N nodes). Level-3 lifts the ORTHOGONAL ceiling inside a single gate: one
criterion may name ≥2 response fields on ONE route, and box-2 certifies them all in ONE
test on ONE fetch = ONE node. This is NOT a multi-gate: ``verify.py`` is untouched, the
spec stays single-node (no ``red_test_nodes`` key), and a single-field criterion is
byte-identical to level-1/2.

These tests pin the synthesiser and derivation against a tiny FastAPI codebase + a disk
fake (no model, no FlowGate). The load-bearing cases: (a) a literal + ≥2 fields still
declines (never-guess trust basis), (b) a multi-field spec drives ``verify`` as a SINGLE
node, (c) a single-field criterion carries no ``asserts`` — the level-1 shape is preserved.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import acceptance_synth as acc
from hive import specify
from hive import verify
from hive.config import RunnerConfig


# ── on-disk codebase (mirrors test_acceptance_synth) ─────────────────────────
def _codebase(tmp_path, *, body_return='{"total": _n(), "items": _rows()}'):
    """GET /api/v1/summary returning a flat sibling dict + a conftest TestClient."""
    root = tmp_path / "code"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1")\n\n'
        '@router.get("/summary")\n'
        "def summary():\n"
        f"    return {body_return}\n",
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


# ── derivation: ≥2 top-level fields → per-field asserts, no bogus container ───
def test_prose_multifield_flat_siblings(tmp_path):
    root = _codebase(tmp_path)
    prose = "GET /api/v1/summary must return non-empty `total` and `items`"
    s = acc.derive_contract_from_prose(prose, root, "AC1")
    assert s is not None and s.kind == "http_read"
    # flat siblings: each field is a TOP-LEVEL key, never nested under a sibling.
    assert [a["json_path"] for a in s.asserts] == ["total", "items"]
    assert all(a["must"] == "non_empty" for a in s.asserts)


def test_prose_multifield_shared_container(tmp_path):
    # leaves genuinely nested in one wrapper keep the shared container.
    root = _codebase(tmp_path, body_return='{"projects": _rows()}')
    (tmp_path / "code" / "app" / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1")\n\n'
        '@router.get("/summary")\n'
        "def summary():\n"
        '    return {"projects": _rows()}\n',
        encoding="utf-8")
    prose = "each project on GET /api/v1/summary must have non-empty `modules` and `tags`"
    s = acc.derive_contract_from_prose(prose, root, "AC1")
    assert s is not None
    assert [a["json_path"] for a in s.asserts] == ["projects[].modules", "projects[].tags"]


def test_prose_multifield_with_literal_declines(tmp_path):
    root = _codebase(tmp_path)
    prose = "GET /api/v1/summary `total` and `items` must be 정확히 3"
    assert acc.derive_contract_from_prose(prose, root, "AC1") is None


def test_single_field_carries_no_asserts(tmp_path):
    # level-1/2 byte-shape: a single field uses the scalar slots, ``asserts`` stays empty.
    root = _codebase(tmp_path)
    prose = "GET /api/v1/summary must return a non-empty `items` field"
    s = acc.derive_contract_from_prose(prose, root, "AC1")
    assert s is not None and s.asserts == ()


# ── explicit oracle: asserts list (per-field must/expected) ──────────────────
def test_explicit_oracle_asserts_list(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/summary",
         "asserts": [{"json_path": "total", "must": "equals", "expected": 3},
                     {"json_path": "items", "must": "non_empty"}]}
    s = acc.validate_explicit_oracle(o, root, "AC2")
    assert s is not None and len(s.asserts) == 2
    assert s.asserts[0]["expected"] == 3 and s.asserts[1]["must"] == "non_empty"


def test_explicit_oracle_single_assert_reduces_to_scalar(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/summary",
         "asserts": [{"json_path": "total", "must": "non_empty"}]}
    s = acc.validate_explicit_oracle(o, root, "AC2")
    # a one-entry list is byte-identical to a plain json_path oracle (no plural slot).
    assert s is not None and s.asserts == () and s.json_path == "total"


def test_explicit_oracle_asserts_rejects_bad_entry(tmp_path):
    root = _codebase(tmp_path)
    o = {"kind": "http_read", "verb": "get", "route": "/api/v1/summary",
         "asserts": [{"json_path": "total", "must": "non_empty"},
                     {"json_path": "items", "must": "equals"}]}  # equals w/o expected
    assert acc.validate_explicit_oracle(o, root, "AC2") is None


# ── synthesis: multi-field is ONE test body, ONE node, ONE file ──────────────
def test_multifield_body_has_all_asserts(tmp_path):
    root = _codebase(tmp_path)
    sym = acc.AcceptanceSymptom(
        kind="http_read", source_ac_id="AC1", verb="get",
        full_path="/api/v1/summary", must="non_empty",
        asserts=({"json_path": "total", "must": "non_empty", "expected": None},
                 {"json_path": "items", "must": "non_empty", "expected": None}))
    res = acc.synthesize_acceptance_red_test(sym, root, app_fixture="client",
                                             test_dir="tests")
    assert res is not None
    content = res["edit"]["content"]
    # one fetch, two field assertions, single def.
    assert content.count("response = client.get(") == 1
    assert "'total' in payload" in content and "'items' in payload" in content
    assert content.count("def test_acceptance_multifield") == 1


def test_specify_multifield_is_single_gate(tmp_path):
    root = _codebase(tmp_path)
    design = (
        "# Feature\n\n"
        "## 수용기준\n"
        "- id: AC1\n"
        "  prose: GET /api/v1/summary must return non-empty `total` and `items`\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    verb: get\n"
        "    route: /api/v1/summary\n"
        "    asserts:\n"
        "    - json_path: total\n"
        "      must: non_empty\n"
        "    - json_path: items\n"
        "      must: non_empty\n"
    )
    spec = {"edits": [{"id": "E1", "file": "app/routes.py",
                       "anchor_old": "_rows()", "replacement_new": "_rows()"}]}
    spec = specify._synthesize_acceptance_red_test(spec, design, root, app_fixture="client")
    v = spec["verify"]
    assert v.get("red_test_node")            # a gate was wired
    assert "red_test_nodes" not in v         # ONE gate, not a multi-gate (level-3 ≠ level-2)
    accept = [e for e in spec["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    assert len(accept) == 1                  # ONE test file
    assert "'total' in payload" in accept[0]["content"]
    assert "'items' in payload" in accept[0]["content"]


# ── verify: a multi-field spec drives ONE node red→green (verify.py unchanged) ─
def _single_node_spec():
    return {
        "_spec_path": "multifield",
        "edits": [
            {"id": "S", "file": "api.py",
             "anchor_old": 'STATE = "buggy"\n', "replacement_new": 'STATE = "fixed"\n'},
            {"id": "T", "kind": "create_file", "file": "tests/test_mf.py",
             "content": "from api import STATE\n\n\ndef test_mf():\n"
                        "    assert STATE == 'fixed'  # stands in for a 2-field assert\n"},
        ],
        "verify": {"red_test_node": "tests/test_mf.py::test_mf", "test_edit_ids": ["T"]},
    }


def _disk_fake(root):
    def fake(command, cwd, node, timeout_sec, env):
        try:
            text = open(os.path.join(root, "api.py"), encoding="utf-8").read()
        except OSError:
            return {"status": verify.RUN_ERROR, "passed": False, "returncode": 2,
                    "raw": "", "cmd": list(command)}
        rc = 0 if "fixed" in text else 1
        return {"status": verify.classify_returncode(rc), "passed": rc == 0,
                "returncode": rc, "raw": "", "cmd": list(command)}
    return fake


def test_multifield_spec_drives_single_node_red_to_green(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    (root / "api.py").write_text('STATE = "buggy"\n', encoding="utf-8")
    (root / "tests").mkdir()
    v = verify.verify_red_green(
        _single_node_spec(), str(root),
        RunnerConfig(command=["pytest", "-q"], cwd="", timeout_sec=30),
        str(tmp_path / "bk"), run_node=_disk_fake(str(root)))
    assert v["transition"] == verify.T_RED_TO_GREEN and v["verified"] is True
    # single-gate contract: no per-gate plural runs.
    assert "red_runs" not in v and "green_runs" not in v
