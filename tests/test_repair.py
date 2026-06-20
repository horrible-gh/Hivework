"""Tests for hive.repair — the self-repair loop that closes the verify gate.

The loop is pure and deterministic: given a sequence of verify verdicts and a
regenerate callback, its stop reason, iteration count, and the spec it returns are
fully determined. So most cases inject a FAKE ``verify_fn`` (a small lookup from a
candidate fix → a scripted transition) and a FAKE ``regenerate`` — no disk, no
model — and assert the gate's contract directly. Two cases drive the REAL
``verify_red_green`` against an on-disk codebase to prove the loop closes a
still-red into a verified green end to end, and that the default specify-backed
regenerator feeds the failing output back as evidence.

The contract under test (every assertion ladders up to this):
  * monotonic — the loop never returns a verified=True it did not observe, and with
    no regenerator it is exactly a single verify (the floor it can never sink below);
  * it bails (never busy-loops) on test-side failures, a None/raising regenerate,
    an oscillating fix, or the iteration cap;
  * when it DOES adopt a re-authored fix, that fix — not the original — is returned.
"""
import os
import sys

from hive import repair
from hive.config import RunnerConfig
from hive.verify import (
    T_NO_BITE,
    T_RED_TO_GREEN,
    T_SKIPPED,
    T_SOURCE_UNAPPLICABLE,
    T_STILL_RED,
)


# --- fakes ------------------------------------------------------------------

def _spec(fix, *, test="assert VALUE == 'fixed'", rationale="r"):
    """A minimal spec keyed on its source replacement (``fix``) so a fake verify
    can map fix → outcome. ``test`` / ``rationale`` vary only the non-fix parts."""
    return {
        "_spec_path": "repair_test_spec",
        "rationale": rationale,
        "edits": [
            {"id": "E1", "file": "app.py", "anchor_old": "OLD\n",
             "replacement_new": fix},
            {"id": "E2", "kind": "create_file", "file": "tests/test_v.py",
             "content": test},
        ],
        "verify": {"red_test_node": "tests/test_v.py::test_v", "test_edit_ids": ["E2"]},
    }


def _make_verify(table, *, calls=None):
    """Fake verify_fn: look the candidate's source fix up in ``table`` → transition.

    Records every call into ``calls`` (if given) so tests can bound how many verifies
    actually ran. A green run carries a captured ``raw`` so the feedback path has
    something to echo.
    """
    def vf(spec, root, runner, backup_root, ttl_hours):
        if calls is not None:
            calls.append(spec["edits"][0]["replacement_new"])
        transition = table[spec["edits"][0]["replacement_new"]]
        return {
            "transition": transition,
            "verified": transition == T_RED_TO_GREEN,
            "node": "tests/test_v.py::test_v",
            "red": {"status": "fail", "raw": "red baseline failed (good)"},
            "green": {"status": "pass" if transition == T_RED_TO_GREEN else "fail",
                      "raw": "" if transition == T_RED_TO_GREEN
                             else "E   AssertionError: VALUE == 'fixed'"},
            "reason": f"scripted {transition}",
        }
    return vf


# --- the monotonic floor ----------------------------------------------------

def test_immediate_green_zero_iterations():
    spec = _spec("VALUE = 'fixed'\n")
    calls = []
    regen_calls = []

    def regen(s, v):
        regen_calls.append(s)
        return _spec("other\n")

    out = repair.repair_red_green(
        spec, "root", _runner(), "bk", regen,
        verify_fn=_make_verify({"VALUE = 'fixed'\n": T_RED_TO_GREEN}, calls=calls))

    assert out["repair_stop"] == repair.STOP_VERIFIED
    assert out["verified"] is True
    assert out["repair_iterations"] == 0
    assert out["repair_history"] == [T_RED_TO_GREEN]
    assert calls == ["VALUE = 'fixed'\n"]      # exactly one verify
    assert regen_calls == []                    # regenerate never needed
    assert out["spec"] is spec                  # original fix returned unchanged


def test_no_regenerator_is_a_single_verify():
    # The floor: with regenerate=None the loop is identical to one verify_red_green.
    spec = _spec("VALUE = 'wrong'\n")
    calls = []
    out = repair.repair_red_green(
        spec, "root", _runner(), "bk", None,
        verify_fn=_make_verify({"VALUE = 'wrong'\n": T_STILL_RED}, calls=calls))
    assert out["repair_stop"] == repair.STOP_NO_REGENERATOR
    assert out["verified"] is False
    assert out["repair_iterations"] == 0
    assert calls == ["VALUE = 'wrong'\n"]       # no extra work, no regeneration


# --- the loop earns its keep -------------------------------------------------

def test_still_red_then_regenerate_to_green():
    bad, good = "VALUE = 'wrong'\n", "VALUE = 'fixed'\n"
    calls = []
    table = {bad: T_STILL_RED, good: T_RED_TO_GREEN}

    seen_verdict = {}

    def regen(spec, verdict):
        seen_verdict.update(verdict)            # the loop must hand the failing verdict on
        return _spec(good)

    out = repair.repair_red_green(
        _spec(bad), "root", _runner(), "bk", regen, max_iters=3,
        verify_fn=_make_verify(table, calls=calls))

    assert out["repair_stop"] == repair.STOP_VERIFIED
    assert out["verified"] is True
    assert out["repair_iterations"] == 1
    assert out["repair_history"] == [T_STILL_RED, T_RED_TO_GREEN]
    assert calls == [bad, good]                 # re-verified the re-authored fix
    assert out["spec"]["edits"][0]["replacement_new"] == good   # the GREEN fix is returned
    # The regenerate saw the real failing test output to work from.
    assert "AssertionError" in seen_verdict["green"]["raw"]


# --- bail paths (never busy-loop) -------------------------------------------

def test_test_side_failure_bails_without_regenerating():
    # A test that does not bite is a TEST problem; re-authoring the source can't fix it.
    spec = _spec("VALUE = 'wrong'\n")
    regen_calls = []
    out = repair.repair_red_green(
        spec, "root", _runner(), "bk", lambda s, v: regen_calls.append(1),
        verify_fn=_make_verify({"VALUE = 'wrong'\n": T_NO_BITE}))
    assert out["repair_stop"] == repair.STOP_UNREPAIRABLE
    assert out["repair_iterations"] == 0
    assert regen_calls == []                     # the costly call was NOT spent


def test_skipped_bails_unrepairable():
    spec = _spec("x\n")
    out = repair.repair_red_green(
        spec, "root", _runner(), "bk", lambda s, v: _spec("y\n"),
        verify_fn=_make_verify({"x\n": T_SKIPPED}))
    assert out["repair_stop"] == repair.STOP_UNREPAIRABLE


def test_regenerate_none_bails():
    spec = _spec("VALUE = 'wrong'\n")
    out = repair.repair_red_green(
        spec, "root", _runner(), "bk", lambda s, v: None,
        verify_fn=_make_verify({"VALUE = 'wrong'\n": T_STILL_RED}))
    assert out["repair_stop"] == repair.STOP_REGEN_EXHAUSTED
    assert out["verified"] is False
    assert out["spec"] is spec                   # the un-verified nothing is not adopted


def test_regenerate_raising_bails_without_crashing():
    spec = _spec("VALUE = 'wrong'\n")

    def boom(s, v):
        raise RuntimeError("author exploded")

    out = repair.repair_red_green(
        spec, "root", _runner(), "bk", boom,
        verify_fn=_make_verify({"VALUE = 'wrong'\n": T_STILL_RED}))
    assert out["repair_stop"] == repair.STOP_REGEN_EXHAUSTED
    assert out["verified"] is False


def test_oscillation_guard_on_repeated_fix():
    # A model that re-proposes a fix it already tried is thrashing — stop, don't re-verify.
    bad = "VALUE = 'wrong'\n"
    calls = []
    out = repair.repair_red_green(
        _spec(bad), "root", _runner(), "bk",
        lambda s, v: _spec(bad, rationale="reworded but SAME fix"),  # same source edit
        max_iters=5, verify_fn=_make_verify({bad: T_STILL_RED}, calls=calls))
    assert out["repair_stop"] == repair.STOP_OSCILLATION
    assert calls == [bad]                         # the repeat was caught BEFORE a 2nd verify


def test_max_iters_exhaustion_is_bounded():
    # Every candidate is distinct but still red → stop exactly at the cap.
    calls = []
    counter = {"n": 0}

    def regen(s, v):
        counter["n"] += 1
        return _spec(f"VALUE = 'try{counter['n']}'\n")

    # All fixes (the initial + every re-author) map to still_red via a defaulting table.
    class _AllRed(dict):
        def __getitem__(self, k):
            return T_STILL_RED

    out = repair.repair_red_green(
        _spec("VALUE = 'try0'\n"), "root", _runner(), "bk", regen,
        max_iters=2, verify_fn=_make_verify(_AllRed(), calls=calls))

    assert out["repair_stop"] == repair.STOP_MAX_ITERS
    assert out["repair_iterations"] == 2
    assert len(calls) == 3                        # initial verify + 2 re-verifies, no more
    assert counter["n"] == 2                       # regenerate called exactly max_iters times


def test_zero_max_iters_does_not_regenerate():
    spec = _spec("VALUE = 'wrong'\n")
    regen_calls = []
    out = repair.repair_red_green(
        spec, "root", _runner(), "bk", lambda s, v: regen_calls.append(1) or _spec("z\n"),
        max_iters=0, verify_fn=_make_verify({"VALUE = 'wrong'\n": T_STILL_RED}))
    assert out["repair_stop"] == repair.STOP_MAX_ITERS
    assert out["repair_iterations"] == 0
    assert regen_calls == []


# --- fingerprint semantics ---------------------------------------------------

def test_fingerprint_ignores_test_edit_and_rationale():
    a = _spec("VALUE = 'fixed'\n", test="t1", rationale="reason A")
    b = _spec("VALUE = 'fixed'\n", test="DIFFERENT TEST", rationale="reason B")
    # Same source fix, different test edit + rationale → same fingerprint.
    assert repair._spec_fingerprint(a) == repair._spec_fingerprint(b)


def test_fingerprint_differs_on_source_change():
    a = _spec("VALUE = 'fixed'\n")
    b = _spec("VALUE = 'other'\n")
    assert repair._spec_fingerprint(a) != repair._spec_fingerprint(b)


# --- feedback rendering ------------------------------------------------------

def test_feedback_block_carries_output_and_prior_edits():
    verdict = {
        "transition": T_STILL_RED, "reason": "the edit is inert",
        "spec": _spec("VALUE = 'wrong'\n"),
        "green": {"status": "fail", "raw": "E  AssertionError: expected 'fixed'"},
    }
    block = repair._render_verify_feedback(verdict, attempt=1)
    assert "repair attempt 1" in block
    assert "still_red" in block
    assert "app.py" in block                       # the prior, inert source edit is named
    assert "VALUE = 'wrong'" in block               # ...and quoted
    assert "AssertionError: expected 'fixed'" in block   # the exact test output is fed back


# --- the default specify-backed regenerator ---------------------------------

def test_specify_regenerator_appends_feedback_and_calls_specify(tmp_path):
    honey = tmp_path / "h.honey.md"
    honey.write_text("ORIGINAL HONEY EVIDENCE\n", encoding="utf-8")
    captured = {}

    def fake_run_specify(*, honey_path, codebase_root, output_path, **kw):
        captured["honey_text"] = open(honey_path, encoding="utf-8").read()
        captured["output_path"] = output_path
        captured["kw"] = kw
        return _spec("VALUE = 'fixed'\n")

    regen = repair.make_specify_regenerator(
        str(honey), "code_root", str(tmp_path / "out.edit_spec.json"),
        run_specify_fn=fake_run_specify, model="m", provider="p")

    verdict = {"transition": T_STILL_RED, "reason": "inert",
               "green": {"raw": "E AssertionError"}}
    new_spec = regen(_spec("VALUE = 'wrong'\n"), verdict)

    assert new_spec["edits"][0]["replacement_new"] == "VALUE = 'fixed'\n"
    # The original honey is preserved and the failing output is appended as feedback.
    assert "ORIGINAL HONEY EVIDENCE" in captured["honey_text"]
    assert "VERIFY FEEDBACK" in captured["honey_text"]
    assert "AssertionError" in captured["honey_text"]
    assert captured["kw"]["model"] == "m" and captured["kw"]["provider"] == "p"


def test_specify_regenerator_returns_none_on_failure(tmp_path):
    honey = tmp_path / "h.md"
    honey.write_text("evidence\n", encoding="utf-8")

    def boom(**kw):
        raise ValueError("specify failed")

    regen = repair.make_specify_regenerator(
        str(honey), "code_root", str(tmp_path / "o.json"), run_specify_fn=boom)
    assert regen(_spec("x\n"), {"transition": T_STILL_RED, "green": {"raw": ""}}) is None


# --- end to end with the REAL verify on disk --------------------------------

def _runner():
    return RunnerConfig(command=["pytest", "-q"], cwd="", timeout_sec=30)


def _disk_spec(fix):
    return {
        "_spec_path": "e2e",
        "edits": [
            {"id": "E1", "file": "app.py",
             "anchor_old": 'VALUE = "buggy"\n', "replacement_new": fix},
            {"id": "E2", "kind": "create_file", "file": "tests/test_value.py",
             "content": "from app import VALUE\n\n\ndef test_value():\n"
                        "    assert VALUE == 'fixed'\n"},
        ],
        "verify": {"red_test_node": "tests/test_value.py::test_value",
                   "test_edit_ids": ["E2"]},
    }


def test_end_to_end_still_red_then_repaired_green(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    (root / "app.py").write_text('VALUE = "buggy"\n', encoding="utf-8")
    (root / "tests").mkdir()
    backup_root = str(tmp_path / "bk")

    def fake_run(command, cwd, node, timeout_sec, env):
        text = open(os.path.join(str(root), "app.py"), encoding="utf-8").read()
        rc = 0 if 'VALUE = "fixed"' in text else 1
        from hive import verify as vmod
        return {"status": vmod.classify_returncode(rc), "passed": rc == 0,
                "returncode": rc, "raw": "" if rc == 0 else "AssertionError", "cmd": []}

    from hive.verify import verify_red_green

    def real_verify(spec, cb, runner, bk, ttl):
        return verify_red_green(spec, cb, runner, bk, run_node=fake_run)

    # The initial fix is WRONG (still red); the regenerator hands back the right fix.
    regen_calls = []

    def regen(spec, verdict):
        regen_calls.append(verdict["transition"])
        return _disk_spec('VALUE = "fixed"\n')

    out = repair.repair_red_green(
        _disk_spec('VALUE = "stillbuggy"\n'), str(root), _runner(), backup_root,
        regen, max_iters=2, verify_fn=real_verify)

    assert out["repair_stop"] == repair.STOP_VERIFIED
    assert out["verified"] is True
    assert out["repair_iterations"] == 1
    assert regen_calls == [T_STILL_RED]
    # Dry run: the loop restored the tree each pass — live source is untouched.
    assert open(str(root / "app.py"), encoding="utf-8").read() == 'VALUE = "buggy"\n'
    assert not (root / "tests" / "test_value.py").exists()


def test_end_to_end_unfixable_stays_red_and_is_monotonic(tmp_path):
    # The regenerator can never produce a green fix → the loop ends no worse than the
    # single-shot still_red it started from (monotonic floor), and never crashes.
    root = tmp_path / "code"
    root.mkdir()
    (root / "app.py").write_text('VALUE = "buggy"\n', encoding="utf-8")
    (root / "tests").mkdir()

    def fake_run(command, cwd, node, timeout_sec, env):
        text = open(os.path.join(str(root), "app.py"), encoding="utf-8").read()
        rc = 0 if 'VALUE = "fixed"' in text else 1
        from hive import verify as vmod
        return {"status": vmod.classify_returncode(rc), "passed": rc == 0,
                "returncode": rc, "raw": "" if rc == 0 else "AssertionError", "cmd": []}

    from hive.verify import verify_red_green

    def real_verify(spec, cb, runner, bk, ttl):
        return verify_red_green(spec, cb, runner, bk, run_node=fake_run)

    n = {"i": 0}

    def regen(spec, verdict):
        n["i"] += 1
        return _disk_spec(f'VALUE = "wrong{n["i"]}"\n')   # always wrong

    out = repair.repair_red_green(
        _disk_spec('VALUE = "wrong0"\n'), str(root), _runner(), str(tmp_path / "bk"),
        regen, max_iters=2, verify_fn=real_verify)

    assert out["repair_stop"] == repair.STOP_MAX_ITERS
    assert out["verified"] is False               # never claimed a green it didn't see
    assert out["transition"] == T_STILL_RED
    assert open(str(root / "app.py"), encoding="utf-8").read() == 'VALUE = "buggy"\n'
