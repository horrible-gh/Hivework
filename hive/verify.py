"""Runtime verify stage — the closed loop that runs a red test red→green.

Why this module exists
----------------------
Every other ``subprocess`` in ``hive/`` calls a *worker* (decompose / retriever /
fan-out), ``git`` (commit), or enforces a *timeout* (providers). NONE of them runs
the TARGET codebase to check that a fix actually killed the symptom. The edit-spec
even carries ``gate.commands`` — but those are only *printed* for a human to run by
hand (``hive/apply.py`` renders them, never executes them). So "apply succeeded"
has always meant "the anchor landed", never "the bug is dead". The result is the
same bug re-investigated round after round, each round closed by a person booting
the whole app to reproduce it manually.

This module is the missing connecting line between *a bug* and *a failing test*.
The materials are already in the edit-spec: ``rationale`` says what the fix must
achieve and ``anchor_old`` / ``replacement_new`` are the red (buggy) and green
(fixed) states. A specify-authored RED TEST (an edit into the target's own
``tests/`` tree) encodes that rationale as one narrow assertion. We then OBSERVE
its transition:

  1. apply ONLY the test edit onto live (un-fixed) code  → run it → must be RED.
  2. apply the source fix on top                          → run it → must be GREEN.
  3. restore everything (this is a dry run; the real write is apply --write).

The trust anchor is step 1. A specify-authored test is a model artifact and can be
wrong in the SAME way the fix is wrong — a test that passes WITHOUT the fix proves
nothing. So a test that is already green before the fix is REJECTED as
non-biting; only an observed red→green transition certifies the fix by execution
rather than by the model's word. This is deliberately the same posture as the rest
of Hivework: we trust a deterministic, environment-grounded observation, not a
worker's self-assessment.

The test runner is NEUTRAL by design (mirrors ``config.DbConnection``): a per-
codebase command + cwd, no FlowGate (or any caller) semantics baked in. When a run's
codebase has no configured runner, verification is SKIPPED gracefully — exactly the
graceful no-op the converge data-read takes when a codebase has no DB entry.
"""

import logging
import os
import subprocess
from typing import Any, Callable

from hive import backup as backup_store
# Reuse apply's byte-preserving IO + test-path convention so the dry-run writes a
# file exactly the way a real apply --write would (EOL/BOM preserved), and we
# partition test vs source edits by the SAME rule apply uses for partial atomicity.
from hive.apply import (
    _encode_preserving,
    _is_test_file,
    _norm_nl,
    _read_text_preserving,
)

logger = logging.getLogger("hive.verify")

# --- run_test_node statuses (one test invocation) ---------------------------
RUN_PASS = "pass"            # exit 0 — assertions held (green)
RUN_FAIL = "fail"            # exit 1 — assertions failed (a legitimate red)
RUN_NO_TESTS = "no_tests"    # pytest exit 5 — nothing collected (proves nothing)
RUN_ERROR = "error"          # other non-zero — collection/import/usage error
RUN_TIMEOUT = "timeout"      # the runner exceeded its timeout

# --- verify_red_green transitions (the overall verdict) ---------------------
T_RED_TO_GREEN = "red_to_green"            # verified-effective by EXECUTION
T_STILL_RED = "still_red"                  # fix applied but test still fails (inert)
T_NO_BITE = "test_does_not_bite"           # test passes WITHOUT the fix (untrustworthy)
T_RED_INDETERMINATE = "red_indeterminate"  # could not even establish the red baseline
T_TEST_UNAPPLICABLE = "test_edit_unapplicable"   # the test edit would not apply
T_SOURCE_UNAPPLICABLE = "source_unapplicable"    # the source edit would not apply
T_SKIPPED = "skipped"                      # no runner / no test target → graceful no-op

# Transitions that mean "the fix is verified safe to ship".
VERIFIED_TRANSITIONS = {T_RED_TO_GREEN}

_RAW_CAP = 4000  # keep captured test output bounded in the verdict


def classify_returncode(returncode: int) -> str:
    """Map a process exit code to a run status (pytest convention, generic-safe).

    pytest: 0=all passed, 1=tests failed, 2=usage error, 3=internal error,
    4=cmdline error, 5=no tests collected. A *fail* (1) is the only non-zero we
    treat as a legitimate RED — every other non-zero means the test did not even
    run as intended, so it can neither confirm a red baseline nor a green result.
    """
    if returncode == 0:
        return RUN_PASS
    if returncode == 1:
        return RUN_FAIL
    if returncode == 5:
        return RUN_NO_TESTS
    return RUN_ERROR


def run_test_node(
    command: list[str],
    cwd: str,
    node_id: str | None,
    timeout_sec: int = 300,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``command`` (plus ``node_id`` as a positional arg) in ``cwd``; report pass/fail.

    The node id is appended verbatim as the final argument — pytest, unittest -k
    targets, and most runners accept a positional selector, which keeps the runner
    config framework-neutral. Returns a result dict: ``status`` (one of the
    ``RUN_*`` constants), ``passed`` (bool, True only on exit 0), ``returncode``,
    a truncated ``raw`` (stdout+stderr) and the resolved ``cmd``. Never raises:
    a timeout or spawn failure is reported as a status, not an exception, so the
    caller's red→green orchestration always reaches its restore step.
    """
    cmd = list(command) + ([node_id] if node_id else [])
    run_env = {**os.environ, **(env or {})}
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout_sec, env=run_env,
        )
    except subprocess.TimeoutExpired as e:
        raw = (e.stdout or "") + (e.stderr or "")
        return {"status": RUN_TIMEOUT, "passed": False, "returncode": None,
                "raw": raw[-_RAW_CAP:], "cmd": cmd}
    except OSError as e:
        return {"status": RUN_ERROR, "passed": False, "returncode": None,
                "raw": f"could not launch test runner: {e}", "cmd": cmd}
    raw = (proc.stdout or "") + (proc.stderr or "")
    status = classify_returncode(proc.returncode)
    return {"status": status, "passed": status == RUN_PASS,
            "returncode": proc.returncode, "raw": raw[-_RAW_CAP:], "cmd": cmd}


def _partition_edits(spec: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Split a spec's edits into (test edits, source edits).

    An explicit ``verify.test_edit_ids`` wins; otherwise we fall back to the path
    convention apply already uses (``_is_test_file``) so the test that asserts the
    fix and the source that implements it are partitioned consistently across
    stages. An edit listed in neither stays a source edit (conservative: it ships
    only after the source phase, never as part of the red baseline).
    """
    edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
    verify = spec.get("verify") if isinstance(spec.get("verify"), dict) else {}
    explicit = {str(x) for x in (verify.get("test_edit_ids") or [])}
    test_edits, source_edits = [], []
    for e in edits:
        eid = str(e.get("id", "?"))
        is_test = eid in explicit if explicit else _is_test_file(e.get("file", ""))
        (test_edits if is_test else source_edits).append(e)
    return test_edits, source_edits


def _write_subset(edits: list[dict], codebase_root: str) -> tuple[bool, str]:
    """Apply a subset of edits to disk (no backup — the caller owns the snapshot).

    Mirrors ``apply.write_edits``' core (byte-preserving anchor replace + create_file)
    but without its own bundle: ``verify_red_green`` takes ONE master snapshot over
    every touched file and guarantees a restore in ``finally``, so a per-call backup
    here would be redundant. Re-verifies anchor uniqueness at write time; the first
    edit that will not apply cleanly stops the subset (caller restores).
    """
    for edit in edits:
        rel = edit.get("file", "")
        if not rel:
            return False, "edit has no 'file'"
        abs_path = os.path.join(codebase_root, rel)
        if edit.get("kind", "edit") == "create_file":
            if os.path.exists(abs_path):
                return False, f"{edit.get('id', '?')}: create_file target exists"
            content = edit.get("content", "")
            if not content or not content.strip():
                return False, f"{edit.get('id', '?')}: create_file content empty"
            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
            with open(abs_path, "wb") as f:
                f.write(content.encode("utf-8"))
            continue
        if not os.path.isfile(abs_path):
            return False, f"{edit.get('id', '?')}: file not found {rel}"
        anchor_old = _norm_nl(edit.get("anchor_old", ""))
        replacement_new = _norm_nl(edit.get("replacement_new", ""))
        text, eol, had_bom = _read_text_preserving(abs_path)
        occurrences = text.count(anchor_old) if anchor_old else 0
        if occurrences != 1:
            return False, (f"{edit.get('id', '?')}: anchor not unique "
                           f"({occurrences} matches) in {rel}")
        modified = text.replace(anchor_old, replacement_new, 1)
        with open(abs_path, "wb") as f:
            f.write(_encode_preserving(modified, eol, had_bom))
    return True, ""


def _touched_paths(edits: list[dict]) -> tuple[list[str], list[str]]:
    """Return (modified rel paths, created rel paths) across ``edits`` for snapshotting."""
    rel_paths: list[str] = []
    created: list[str] = []
    for e in edits:
        rel = e.get("file", "")
        if not rel:
            continue
        if e.get("kind", "edit") == "create_file":
            if rel not in created:
                created.append(rel)
        elif rel not in rel_paths:
            rel_paths.append(rel)
    return rel_paths, created


def verify_red_green(
    spec: dict[str, Any],
    codebase_root: str,
    runner: Any,
    backup_root: str,
    ttl_hours: int = 168,
    run_node: Callable[..., dict[str, Any]] = run_test_node,
) -> dict[str, Any]:
    """Observe a spec's red test go red→green, then restore (a non-destructive dry run).

    Sequence (all under ONE master snapshot, restored in ``finally``):
      1. write ONLY the test edit(s) onto the live, un-fixed code,
      2. run the red-test node — it MUST fail (RED). A pass here means the test does
         not bite (``test_does_not_bite``); an error/no-collection is indeterminate,
      3. write the source fix edit(s) on top,
      4. run the node again — it MUST pass (GREEN); else the fix is inert (``still_red``),
      5. restore every touched file from the master snapshot.

    ``runner`` is the resolved test-runner config (``config.RunnerConfig``);
    ``run_node`` is injectable so tests can drive the orchestration with a fake
    runner. Returns a verdict dict with ``transition`` (a ``T_*`` constant),
    ``verified`` (bool), the ``node``, and the captured ``red`` / ``green`` runs.
    """
    verify_block = spec.get("verify") if isinstance(spec.get("verify"), dict) else {}
    node = verify_block.get("red_test_node") or ""
    test_edits, source_edits = _partition_edits(spec)

    verdict: dict[str, Any] = {
        "transition": T_SKIPPED, "verified": False, "node": node,
        "red": None, "green": None, "reason": "",
    }

    if runner is None:
        verdict["reason"] = "no test runner configured for this codebase"
        return verdict
    if not node:
        verdict["reason"] = "spec has no verify.red_test_node to run"
        return verdict
    if not test_edits:
        verdict["reason"] = "spec carries no test edit to establish a red baseline"
        return verdict

    cwd = runner.cwd if os.path.isabs(runner.cwd or "") else os.path.join(
        codebase_root, runner.cwd or "")
    rel_paths, created = _touched_paths(test_edits + source_edits)

    os.makedirs(backup_root, exist_ok=True)
    backup_store.purge_expired(backup_root, fallback_ttl_hours=ttl_hours)
    bundle = backup_store.create_bundle(
        backup_root, spec.get("_spec_path", "verify"), codebase_root,
        rel_paths, ttl_hours, created_paths=created)
    verdict["bundle"] = bundle["dir"]

    def _run() -> dict[str, Any]:
        return run_node(runner.command, cwd, node, runner.timeout_sec,
                        getattr(runner, "env", None) or None)

    try:
        # 1+2: red baseline — test edit only, no fix yet.
        ok, why = _write_subset(test_edits, codebase_root)
        if not ok:
            verdict["transition"] = T_TEST_UNAPPLICABLE
            verdict["reason"] = why
            return verdict
        red = _run()
        verdict["red"] = red
        if red["status"] == RUN_PASS:
            verdict["transition"] = T_NO_BITE
            verdict["reason"] = ("red test passed WITHOUT the fix — it does not "
                                 "reproduce the symptom and cannot certify a fix")
            return verdict
        if red["status"] != RUN_FAIL:
            verdict["transition"] = T_RED_INDETERMINATE
            verdict["reason"] = (f"red baseline did not run cleanly "
                                 f"(status={red['status']}) — cannot certify")
            return verdict

        # 3+4: apply the fix, expect green.
        ok, why = _write_subset(source_edits, codebase_root)
        if not ok:
            verdict["transition"] = T_SOURCE_UNAPPLICABLE
            verdict["reason"] = why
            return verdict
        green = _run()
        verdict["green"] = green
        if green["status"] == RUN_PASS:
            verdict["transition"] = T_RED_TO_GREEN
            verdict["verified"] = True
            verdict["reason"] = "red test failed without the fix and passes with it"
        else:
            verdict["transition"] = T_STILL_RED
            verdict["reason"] = (f"fix applied but the red test still does not pass "
                                 f"(status={green['status']}) — the edit is inert")
        return verdict
    finally:
        # The dry run never persists: put every touched file back exactly.
        try:
            backup_store.restore_bundle(bundle["dir"])
        except Exception as e:  # restore must never mask the verdict
            logger.error("verify: restore failed for %s: %s", bundle["dir"], e)
            verdict["restore_failed"] = True
