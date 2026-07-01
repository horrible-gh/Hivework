"""box-1 (group 0066, level-2): multiple acceptance criteria → multiple red→green gates.

Level-1 (group 0064/0065) synthesised at most ONE acceptance gate per feature — specify
returned at the first resolving ``## 수용기준`` item and ``verify`` ran a single
``red_test_node``. A design that lists N criteria therefore certified only one of them.

box-1 lifts that ceiling WITHOUT breaking the single-gate contract:

  - ``specify._synthesize_acceptance_red_test`` now accumulates EVERY resolving criterion
    into its own RED test file (unique id/slug), registering the first as ``red_test_node``
    (backward compat) and the full list as ``red_test_nodes``.
  - ``verify.verify_red_green`` runs the whole node list: all must bite (red) and all must
    pass after the fix (green); one non-biting or one still-red gate fails the run.

These tests pin both halves against fakes/disk — no model, no FlowGate — and assert that a
SINGLE resolving criterion still produces the byte-identical level-1 shape (no
``red_test_nodes`` key), so the 0064/0065 behaviour is preserved.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import specify
from hive import verify
from hive.config import RunnerConfig


# ── design fixtures ──────────────────────────────────────────────────────────
_TWO_CRITERIA = (
    "# Feature\n\n"
    "## 수용기준\n"
    "- id: AC1\n"
    "  prose: app/calc.py::answer must equal 42\n"
    "  oracle:\n"
    "    kind: unit_value\n"
    "    target: app/calc.py::answer\n"
    "    must: equals\n"
    "    expected: 42\n"
    "- id: AC2\n"
    "  prose: app/calc.py::greeting must equal 'hi'\n"
    "  oracle:\n"
    "    kind: unit_value\n"
    "    target: app/calc.py::greeting\n"
    "    must: equals\n"
    "    expected: hi\n"
)

_ONE_CRITERION = (
    "# Feature\n\n"
    "## 수용기준\n"
    "- id: AC1\n"
    "  prose: app/calc.py::answer must equal 42\n"
    "  oracle:\n"
    "    kind: unit_value\n"
    "    target: app/calc.py::answer\n"
    "    must: equals\n"
    "    expected: 42\n"
)


def _sut(tmp_path):
    root = tmp_path / "sut"
    (root / "app").mkdir(parents=True)
    # Both symbols exist so detect_acceptance resolves each criterion (RED values here).
    (root / "app" / "calc.py").write_text("answer = 0\ngreeting = 'x'\n", encoding="utf-8")
    return str(root)


def _spec_with_source():
    # A source edit must exist or synthesis no-ops (nothing to certify).
    return {"edits": [{"id": "E1", "file": "app/calc.py",
                       "anchor_old": "answer = 0\n", "replacement_new": "answer = 42\n"}]}


# ── specify: N criteria → N gates ────────────────────────────────────────────
def test_two_criteria_yield_two_gates(tmp_path):
    root = _sut(tmp_path)
    spec = specify._synthesize_acceptance_red_test(
        _spec_with_source(), _TWO_CRITERIA, root)
    v = spec["verify"]
    # Two distinct gate nodes, first is the primary single-gate node.
    assert v["red_test_node"] == v["red_test_nodes"][0]
    assert len(v["red_test_nodes"]) == 2
    # Two independent test files with unique ids.
    accept_edits = [e for e in spec["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    assert len(accept_edits) == 2
    assert len({e["id"] for e in accept_edits}) == 2      # ids unique
    assert len({e["file"] for e in accept_edits}) == 2    # files unique
    assert set(v["test_edit_ids"]) == {e["id"] for e in accept_edits}


def test_single_criterion_stays_level1_shape(tmp_path):
    root = _sut(tmp_path)
    spec = specify._synthesize_acceptance_red_test(
        _spec_with_source(), _ONE_CRITERION, root)
    v = spec["verify"]
    assert v["red_test_node"]
    # Level-1 byte-shape: no plural key when there is a single gate.
    assert "red_test_nodes" not in v
    accept_edits = [e for e in spec["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    assert len(accept_edits) == 1


# ── verify: the whole node list must go red→green ────────────────────────────
def _multi_gate_spec():
    """Two source fixes + two red tests, wired as two gates."""
    return {
        "_spec_path": "multigate",
        "edits": [
            {"id": "S_A", "file": "a.py",
             "anchor_old": 'A = "buggy"\n', "replacement_new": 'A = "fixed"\n'},
            {"id": "S_B", "file": "b.py",
             "anchor_old": 'B = "buggy"\n', "replacement_new": 'B = "fixed"\n'},
            {"id": "T_A", "kind": "create_file", "file": "tests/test_a.py",
             "content": "from a import A\n\n\ndef test_a():\n    assert A == 'fixed'\n"},
            {"id": "T_B", "kind": "create_file", "file": "tests/test_b.py",
             "content": "from b import B\n\n\ndef test_b():\n    assert B == 'fixed'\n"},
        ],
        "verify": {
            "red_test_node": "tests/test_a.py::test_a",
            "red_test_nodes": ["tests/test_a.py::test_a", "tests/test_b.py::test_b"],
            "test_edit_ids": ["T_A", "T_B"],
        },
    }


def _two_file_codebase(tmp_path, *, a="buggy", b="buggy"):
    root = tmp_path / "code"
    root.mkdir()
    (root / "a.py").write_text(f'A = "{a}"\n', encoding="utf-8")
    (root / "b.py").write_text(f'B = "{b}"\n', encoding="utf-8")
    (root / "tests").mkdir()
    return str(root)


def _runner():
    return RunnerConfig(command=["pytest", "-q"], cwd="", timeout_sec=30)


def _disk_fake(root):
    """Pass a gate iff its target file has been fixed on live disk."""
    def fake(command, cwd, node, timeout_sec, env):
        fname = "a.py" if "test_a" in node else "b.py"
        try:
            text = open(os.path.join(root, fname), encoding="utf-8").read()
        except OSError:
            return {"status": verify.RUN_ERROR, "passed": False, "returncode": 2,
                    "raw": "", "cmd": list(command)}
        rc = 0 if "fixed" in text else 1
        return {"status": verify.classify_returncode(rc), "passed": rc == 0,
                "returncode": rc, "raw": "", "cmd": list(command)}
    return fake


def test_all_gates_red_then_green(tmp_path):
    root = _two_file_codebase(tmp_path)
    v = verify.verify_red_green(_multi_gate_spec(), root, _runner(),
                                str(tmp_path / "bk"), run_node=_disk_fake(root))
    assert v["transition"] == verify.T_RED_TO_GREEN
    assert v["verified"] is True
    assert len(v["red_runs"]) == 2 and all(r["status"] == verify.RUN_FAIL for r in v["red_runs"])
    assert len(v["green_runs"]) == 2 and all(g["passed"] for g in v["green_runs"])
    # Dry run restored: both sources reverted, both tests gone.
    assert 'A = "buggy"' in open(os.path.join(root, "a.py"), encoding="utf-8").read()
    assert not os.path.exists(os.path.join(root, "tests", "test_a.py"))


def test_one_gate_still_red_fails_the_run(tmp_path):
    root = _two_file_codebase(tmp_path)
    spec = _multi_gate_spec()
    # Drop the second source fix → gate B never goes green.
    spec["edits"] = [e for e in spec["edits"] if e["id"] != "S_B"]
    v = verify.verify_red_green(spec, root, _runner(),
                                str(tmp_path / "bk"), run_node=_disk_fake(root))
    assert v["transition"] == verify.T_STILL_RED
    assert v["verified"] is False


def test_one_non_biting_gate_poisons_the_run(tmp_path):
    # Gate A's target is ALREADY fixed on disk → it passes without the fix → no bite.
    root = _two_file_codebase(tmp_path, a="fixed")
    v = verify.verify_red_green(_multi_gate_spec(), root, _runner(),
                                str(tmp_path / "bk"), run_node=_disk_fake(root))
    assert v["transition"] == verify.T_NO_BITE
    assert v["verified"] is False
    assert v["green"] is None  # never advanced to green
