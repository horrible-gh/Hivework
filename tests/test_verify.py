"""Tests for hive.verify — the runtime red→green closed-loop gate.

These exercise the orchestration with a FAKE run_node (no pytest / FlowGate needed)
plus real on-disk writes + restore, so the red baseline, the no-bite rejection, the
still-red (inert fix) case, and snapshot restoration are all observed deterministically.
A couple of cases drive the genuine subprocess path with tiny `python -c` commands to
cover exit-code classification end to end.
"""
import json
import os
import sys

from hive import verify
from hive.apply import run_apply
from hive.config import RunnerConfig


def _spec(tmp_path, *, red_node="tests/test_value.py::test_value"):
    """A minimal spec: source fix E1 (buggy→fixed) + red test E2 (create_file)."""
    return {
        "_spec_path": "verify_test_spec",
        "edits": [
            {"id": "E1", "file": "app.py",
             "anchor_old": 'VALUE = "buggy"\n',
             "replacement_new": 'VALUE = "fixed"\n'},
            {"id": "E2", "kind": "create_file", "file": "tests/test_value.py",
             "content": "from app import VALUE\n\n\ndef test_value():\n    assert VALUE == 'fixed'\n"},
        ],
        "verify": {"red_test_node": red_node, "test_edit_ids": ["E2"]},
    }


def _make_codebase(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    (root / "app.py").write_text('VALUE = "buggy"\n', encoding="utf-8")
    (root / "tests").mkdir()
    return str(root)


def _runner():
    return RunnerConfig(command=["pytest", "-q"], cwd="", timeout_sec=30)


def test_classify_returncode():
    assert verify.classify_returncode(0) == verify.RUN_PASS
    assert verify.classify_returncode(1) == verify.RUN_FAIL
    assert verify.classify_returncode(5) == verify.RUN_NO_TESTS
    assert verify.classify_returncode(2) == verify.RUN_ERROR
    assert verify.classify_returncode(3) == verify.RUN_ERROR


def test_run_test_node_real_subprocess(tmp_path):
    # exit 0 → pass, exit 1 → fail, exit 5 → no_tests — real subprocess.
    p = verify.run_test_node([sys.executable, "-c", "import sys; sys.exit(0)"],
                             str(tmp_path), None, timeout_sec=30)
    assert p["status"] == verify.RUN_PASS and p["passed"] is True
    f = verify.run_test_node([sys.executable, "-c", "import sys; sys.exit(1)"],
                             str(tmp_path), None, timeout_sec=30)
    assert f["status"] == verify.RUN_FAIL and f["passed"] is False
    n = verify.run_test_node([sys.executable, "-c", "import sys; sys.exit(5)"],
                             str(tmp_path), None, timeout_sec=30)
    assert n["status"] == verify.RUN_NO_TESTS


def test_run_test_node_bad_command(tmp_path):
    r = verify.run_test_node(["this_command_does_not_exist_zzz"], str(tmp_path),
                             None, timeout_sec=30)
    assert r["status"] == verify.RUN_ERROR and r["passed"] is False


def test_red_to_green_and_restore(tmp_path):
    root = _make_codebase(tmp_path)
    spec = _spec(tmp_path)
    backup_root = str(tmp_path / "backups")

    # Fake runner: read the live app.py and pass iff it already says "fixed".
    def fake_run(command, cwd, node, timeout_sec, env):
        text = open(os.path.join(root, "app.py"), encoding="utf-8").read()
        rc = 0 if "fixed" in text else 1
        return {"status": verify.classify_returncode(rc), "passed": rc == 0,
                "returncode": rc, "raw": "", "cmd": list(command)}

    v = verify.verify_red_green(spec, root, _runner(), backup_root,
                                run_node=fake_run)
    assert v["transition"] == verify.T_RED_TO_GREEN
    assert v["verified"] is True
    assert v["red"]["status"] == verify.RUN_FAIL
    assert v["green"]["status"] == verify.RUN_PASS
    # Dry run: everything restored — source reverted, created test gone.
    assert open(os.path.join(root, "app.py"), encoding="utf-8").read() == 'VALUE = "buggy"\n'
    assert not os.path.exists(os.path.join(root, "tests", "test_value.py"))


def test_test_does_not_bite(tmp_path):
    root = _make_codebase(tmp_path)
    spec = _spec(tmp_path)
    backup_root = str(tmp_path / "backups")

    # Test passes even WITHOUT the fix → non-biting → rejected.
    def always_pass(command, cwd, node, timeout_sec, env):
        return {"status": verify.RUN_PASS, "passed": True, "returncode": 0,
                "raw": "", "cmd": list(command)}

    v = verify.verify_red_green(spec, root, _runner(), backup_root,
                               run_node=always_pass)
    assert v["transition"] == verify.T_NO_BITE
    assert v["verified"] is False
    assert v["green"] is None  # never advanced to the green phase


def test_still_red_inert_fix(tmp_path):
    root = _make_codebase(tmp_path)
    spec = _spec(tmp_path)
    backup_root = str(tmp_path / "backups")

    # Always fails — even after the fix the symptom persists → inert.
    def always_fail(command, cwd, node, timeout_sec, env):
        return {"status": verify.RUN_FAIL, "passed": False, "returncode": 1,
                "raw": "", "cmd": list(command)}

    v = verify.verify_red_green(spec, root, _runner(), backup_root,
                               run_node=always_fail)
    assert v["transition"] == verify.T_STILL_RED
    assert v["verified"] is False


def test_red_indeterminate(tmp_path):
    root = _make_codebase(tmp_path)
    spec = _spec(tmp_path)
    backup_root = str(tmp_path / "backups")

    def no_tests(command, cwd, node, timeout_sec, env):
        return {"status": verify.RUN_NO_TESTS, "passed": False, "returncode": 5,
                "raw": "", "cmd": list(command)}

    v = verify.verify_red_green(spec, root, _runner(), backup_root,
                               run_node=no_tests)
    assert v["transition"] == verify.T_RED_INDETERMINATE


def test_skipped_when_no_runner(tmp_path):
    root = _make_codebase(tmp_path)
    spec = _spec(tmp_path)
    v = verify.verify_red_green(spec, root, None, str(tmp_path / "b"))
    assert v["transition"] == verify.T_SKIPPED


def test_skipped_when_no_test_edit(tmp_path):
    root = _make_codebase(tmp_path)
    spec = _spec(tmp_path)
    spec["edits"] = [spec["edits"][0]]  # drop the test edit
    spec["verify"]["test_edit_ids"] = []
    v = verify.verify_red_green(spec, root, _runner(), str(tmp_path / "b"),
                               run_node=lambda *a, **k: None)
    assert v["transition"] == verify.T_SKIPPED


# --- end-to-end through apply.run_apply with a REAL subprocess runner ---------

_REAL_RUNNER_CODE = (
    "import sys; "
    "txt = open('app.py', encoding='utf-8').read(); "
    "sys.exit(0 if 'fixed' in txt else 1)"
)


def _write_spec(tmp_path, root):
    spec = _spec(tmp_path)
    spec["termination"] = "ready_to_apply"
    spec["codebase_root"] = root
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return str(p)


def test_run_apply_verify_confirms_red_to_green(tmp_path):
    root = _make_codebase(tmp_path)
    spec_path = _write_spec(tmp_path, root)
    runner = RunnerConfig(command=[sys.executable, "-c", _REAL_RUNNER_CODE],
                          cwd="", timeout_sec=30)
    proposal = run_apply(
        spec_path=spec_path, codebase_root=root, write=False, verify=True,
        runner=runner, backup_root=str(tmp_path / "backups"))
    assert proposal["runtime_verify"]["transition"] == verify.T_RED_TO_GREEN
    assert proposal["ready"] is True
    # Dry run only: nothing persisted.
    assert open(os.path.join(root, "app.py"), encoding="utf-8").read() == 'VALUE = "buggy"\n'
    assert not os.path.exists(os.path.join(root, "tests", "test_value.py"))


def test_run_apply_verify_blocks_when_non_biting(tmp_path):
    root = _make_codebase(tmp_path)
    spec_path = _write_spec(tmp_path, root)
    # A runner that always passes → the red test never bites → READY is withheld.
    runner = RunnerConfig(command=[sys.executable, "-c", "import sys; sys.exit(0)"],
                          cwd="", timeout_sec=30)
    proposal = run_apply(
        spec_path=spec_path, codebase_root=root, write=False, verify=True,
        runner=runner, backup_root=str(tmp_path / "backups"))
    assert proposal["runtime_verify"]["transition"] == verify.T_NO_BITE
    assert proposal["ready"] is False
    assert any("runtime verify" in r for r in proposal["not_ready_reasons"])


def test_rebase_node_to_cwd_translates_path(tmp_path):
    # spec references files from codebase_root; a runner with cwd=server must get the
    # node rebased to cwd-relative or pytest looks for server/server/... and errors.
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "server", "tests"))
    with open(os.path.join(root, "server", "tests", "test_x.py"), "w") as f:
        f.write("def test_x():\n    assert True\n")
    cwd = os.path.join(root, "server")
    node = "server/tests/test_x.py::test_x"
    assert verify._rebase_node_to_cwd(node, root, cwd) == "tests/test_x.py::test_x"


def test_rebase_node_leaves_dotted_and_missing_alone(tmp_path):
    root = str(tmp_path)
    cwd = os.path.join(root, "server")
    # dotted unittest id (no path separator) — untouched
    assert verify._rebase_node_to_cwd("pkg.mod.TestY", root, cwd) == "pkg.mod.TestY"
    # a path that does not exist on disk — untouched (never guess)
    assert verify._rebase_node_to_cwd(
        "server/tests/missing.py::t", root, cwd) == "server/tests/missing.py::t"
