"""Self-repair loop — close the verify gate by FEEDING the failing test back to the fixer.

[hive.verify] observes ONE fix go red→green and stops at the first verdict: a
``still_red`` (the fix applied but the red test still fails) just marks the proposal
NOT READY and the same bug is handed back. A human — or a costlier model — then reads
the failing test output and writes a better fix. That read-the-failure-and-retry loop
is exactly what a strong model runs in its head; a cheap model never gets the chance.
This module externalizes the loop so a cheap model reaches the same ``verified_fixed``
result by ITERATION rather than by model tier:

    verify → still_red? → feed the captured test output back to ``regenerate`` → verify → …

Design posture (mirrors [hive.reinvestigate]): the loop itself is PURE and
deterministic — a bounded iteration cap plus an oscillation guard. The ONLY paid
step is the injected ``regenerate`` callback that authors the next candidate fix. It
is a GATE, not a generator: it never invents a verdict, it only re-runs the
deterministic red→green observation on each candidate and ACCEPTS the first that
transitions. The trust anchor stays the test, not the model's word.

That makes the addition **monotonic**: the worst case is the single-shot result it
started from (no verified fix) — never worse. The loop can only turn a ``still_red``
into a verified green or leave the verdict exactly as it was. That is what makes it
safe to bolt onto an unstable foundation: it adds no new failure mode of its own.

Why it bails instead of looping forever:
  - verified (red→green)              → accept this fix, done.
  - the failure is TEST-side          → re-authoring the *source* fix cannot help
    (no_bite / red_indeterminate /       (the test, not the fix, is the problem); bail
     test_unapplicable / skipped)         with the current verdict.
  - ``regenerate`` returns None       → the fixer gave up; bail.
  - the new fix repeats one already   → oscillation: the model is thrashing between the
    tried (compared by fingerprint)     same wrong fixes; further calls only burn budget.
  - ``max_iters`` reached             → honest stop with the best verdict observed.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable

from hive.verify import (
    T_SOURCE_UNAPPLICABLE,
    T_STILL_RED,
    VERIFIED_TRANSITIONS,
    _partition_edits,
    verify_red_green,
)

logger = logging.getLogger("hive.repair")

# Transitions where re-authoring the SOURCE fix is the right lever — the fix is the
# thing at fault. Every OTHER non-verified transition is test-side or environmental
# (the test does not bite, could not establish a red baseline, would not apply, or
# verification was skipped), so producing a new source fix cannot change it. On
# those the loop bails immediately rather than spend a regenerate call that cannot
# move the verdict.
REPAIRABLE_TRANSITIONS = {T_STILL_RED, T_SOURCE_UNAPPLICABLE}

# repair_stop reasons (why the loop ended) -----------------------------------
STOP_VERIFIED = "verified"                      # a candidate transitioned red→green
STOP_UNREPAIRABLE = "unrepairable"              # failure is test-side, not fix-side
STOP_NO_REGENERATOR = "no_regenerator"          # no regenerate callback → single verify
STOP_REGEN_EXHAUSTED = "regenerate_exhausted"   # regenerate returned None / raised
STOP_OSCILLATION = "oscillation"                # a candidate repeated an already-tried fix
STOP_MAX_ITERS = "max_iters"                    # iteration cap hit without a green


def _spec_fingerprint(spec: dict[str, Any]) -> str:
    """Stable fingerprint of a spec's SOURCE edits — the part a re-fix changes.

    The oscillation guard compares candidates by what they would WRITE (file +
    kind + anchor_old + replacement_new/content), not by incidental fields (ids,
    rationale wording, the test edit). Two candidates that touch the same lines
    with the same replacement are the SAME fix even if the model reworded its
    reasoning — so identical wrong fixes are caught as thrashing. The test edit is
    excluded on purpose: it is the RED baseline, held constant across iterations,
    so a changed test must not look like a changed fix.
    """
    _test_edits, source_edits = _partition_edits(spec)
    sig = sorted(
        (
            str(e.get("file", "")),
            str(e.get("kind", "edit")),
            str(e.get("anchor_old", "")),
            str(e.get("replacement_new", "")),
            str(e.get("content", "")),
        )
        for e in source_edits
    )
    blob = json.dumps(sig, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _finish(
    verdict: dict[str, Any],
    spec: dict[str, Any],
    iterations: int,
    history: list[str],
    stop: str,
) -> dict[str, Any]:
    """Augment a verify verdict with the loop's bookkeeping and return it.

    ``spec`` is the candidate the returned verdict belongs to — the verified fix
    when one was reached, otherwise the last fix that was actually verify-run (never
    an un-verified candidate). The caller applies THIS spec, not the original.
    """
    out = dict(verdict)
    out["spec"] = spec
    out["repair_iterations"] = iterations
    out["repair_history"] = list(history)
    out["repair_stop"] = stop
    return out


def repair_red_green(
    spec: dict[str, Any],
    codebase_root: str,
    runner: Any,
    backup_root: str,
    regenerate: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None = None,
    *,
    max_iters: int = 2,
    ttl_hours: int = 168,
    verify_fn: Callable[..., dict[str, Any]] = verify_red_green,
) -> dict[str, Any]:
    """Loop verify→regenerate→verify until the fix is verified or a stop fires.

    Returns the verify verdict dict (carrying a ``T_*`` ``transition``) augmented
    with:
      - ``spec``: the candidate the verdict belongs to — the verified fix if one was
        reached, else the last fix actually verify-run. Apply THIS, not the original.
      - ``repair_iterations``: regenerate→verify rounds run beyond the initial verify
        (0 means the initial spec verified, or the loop bailed before re-authoring).
      - ``repair_history``: the ``transition`` of every verify, in order.
      - ``repair_stop``: one of the ``STOP_*`` constants.

    ``regenerate(spec, verdict) -> spec | None`` authors the next candidate fix from
    the failing verdict (which carries the green run's captured test output under
    ``verdict['green']['raw']``). It is the sole paid step and is injected so the loop
    is unit-testable with a fake. When ``regenerate`` is None the loop degenerates to
    a SINGLE verify — identical to calling ``verify_red_green`` directly — which keeps
    the addition strictly monotonic: turning the loop on can never make a fix worse,
    only let a cheap model iterate its way to a green one.

    ``max_iters`` bounds the regenerate→verify rounds (the cap excludes the initial
    verify). ``verify_fn`` is injectable for the same reason ``regenerate`` is.
    """
    if max_iters < 0:
        max_iters = 0

    current = spec
    history: list[str] = []
    iterations = 0
    # Seed with the original fix so a regenerate that returns an identical candidate
    # is caught as oscillation rather than re-verified.
    seen: set[str] = {_spec_fingerprint(spec)}

    while True:
        verdict = verify_fn(current, codebase_root, runner, backup_root, ttl_hours)
        transition = verdict.get("transition")
        history.append(transition)

        if verdict.get("verified") or transition in VERIFIED_TRANSITIONS:
            if iterations:
                logger.info("repair: VERIFIED after %d repair iteration(s)", iterations)
            return _finish(verdict, current, iterations, history, STOP_VERIFIED)

        # No budget/way to author a different fix → stop at the single-shot floor.
        if regenerate is None:
            return _finish(verdict, current, iterations, history, STOP_NO_REGENERATOR)
        # The failure is not something a new SOURCE fix can move → don't burn a call.
        if transition not in REPAIRABLE_TRANSITIONS:
            return _finish(verdict, current, iterations, history, STOP_UNREPAIRABLE)
        # Iteration cap (excludes the initial verify) → honest stop.
        if iterations >= max_iters:
            logger.info("repair: max_iters=%d reached without a verified fix", max_iters)
            return _finish(verdict, current, iterations, history, STOP_MAX_ITERS)

        logger.info("repair: attempt %d/%d — %s; re-authoring the fix from the test output",
                    iterations + 1, max_iters, transition)
        try:
            revised = regenerate(current, verdict)
        except Exception as e:  # a regenerate hiccup must not lose the standing verdict
            logger.error("repair: regenerate raised on attempt %d: %s", iterations + 1, e)
            return _finish(verdict, current, iterations, history, STOP_REGEN_EXHAUSTED)

        if not revised:
            logger.info("repair: regenerate produced no new fix on attempt %d — bail",
                        iterations + 1)
            return _finish(verdict, current, iterations, history, STOP_REGEN_EXHAUSTED)

        fp = _spec_fingerprint(revised)
        if fp in seen:
            logger.info("repair: re-authored fix repeats an already-tried fix — "
                        "oscillation, bail")
            return _finish(verdict, current, iterations, history, STOP_OSCILLATION)

        seen.add(fp)
        # Carry forward the snapshot-naming path so verify can name its backup bundle.
        if "_spec_path" not in revised and spec.get("_spec_path"):
            revised["_spec_path"] = spec["_spec_path"]
        current = revised
        iterations += 1


# --- default regenerator: re-author the fix through specify ------------------
_RAW_CAP = 4000  # bound the test output fed back into the next author prompt


def _render_verify_feedback(verdict: dict[str, Any], attempt: int) -> str:
    """Render a failing red→green verdict as a feedback block for the next author.

    The previous fix was applied on top of the red test and the test STILL failed
    (or the source edit would not apply). To do better the author needs three things:
    that its last fix was inert, WHICH edits it tried, and the EXACT test output. We
    hand all three back as plain markdown appended to the honey — the same evidence
    channel every other specify input already rides on — so a cheap author writes its
    next fix WITH the red test's own complaint in hand.
    """
    spec = verdict.get("spec") if isinstance(verdict.get("spec"), dict) else {}
    _test_edits, source_edits = _partition_edits(spec) if spec else ([], [])
    lines = [
        f"## VERIFY FEEDBACK (repair attempt {attempt}) — the previous fix did NOT work",
        "",
        f"The red test was kept and your last fix was applied on top of it, but the "
        f"observation was `{verdict.get('transition')}`: {verdict.get('reason', '')}",
        "",
        "Your previous source edit(s) — these are NOT effective, author a DIFFERENT fix "
        "(do not repeat them, and do not change the red test):",
    ]
    if source_edits:
        for e in source_edits:
            lines.append(f"  - {e.get('file', '?')} :: {e.get('id', '?')}")
            repl = (e.get("replacement_new") or e.get("content") or "").strip()
            if repl:
                lines.append("    ```")
                lines.extend("    " + ln for ln in repl.splitlines()[:40])
                lines.append("    ```")
    else:
        lines.append("  (none recorded)")
    # The captured output: prefer the green run (fix applied, still failing); fall
    # back to the red run when the source edit never even applied.
    run = verdict.get("green") or verdict.get("red") or {}
    raw = (run.get("raw") or "").strip()
    if raw:
        lines += [
            "",
            "Exact test output (this is the assertion you must actually satisfy):",
            "```",
            raw[-_RAW_CAP:],
            "```",
        ]
    return "\n".join(lines) + "\n"


def make_specify_regenerator(
    honey_path: str,
    codebase_root: str,
    spec_out: str,
    *,
    run_specify_fn: Callable[..., dict[str, Any]] | None = None,
    **specify_kwargs: Any,
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None]:
    """Build the default ``regenerate(spec, verdict)`` — re-author a fix via specify.

    The returned closure appends the failing verdict (transition + the previous,
    inert edits + the captured test output) to the ORIGINAL honey as a feedback
    section and re-runs specify on the augmented honey. Each attempt writes a distinct
    ``<spec_out>.repairN.json`` (and a sibling ``.repairN.honey.md``) so intermediate
    candidates are inspectable and never clobber the original spec. ``run_specify_fn``
    is injectable; by default it is ``hive.specify.run_specify`` (imported lazily to
    avoid pulling the heavy specify module into every import of this loop).

    A regenerate that cannot produce a spec returns None, which the loop treats as
    ``regenerate_exhausted`` and bails — keeping the whole feature monotonic.
    """
    if run_specify_fn is None:
        from hive.specify import run_specify as run_specify_fn  # lazy: heavy module

    with open(honey_path, "r", encoding="utf-8") as f:
        base_honey = f.read()

    base, ext = (spec_out.rsplit(".", 1) + ["json"])[:2]
    counter = {"n": 0}

    def regenerate(spec: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any] | None:
        counter["n"] += 1
        n = counter["n"]
        # The verdict the loop passes does not carry the failing spec; attach it so the
        # feedback block can quote the inert edits the author must avoid repeating.
        v = dict(verdict)
        v.setdefault("spec", spec)
        feedback = _render_verify_feedback(v, n)
        honey_n = f"{base}.repair{n}.honey.md"
        with open(honey_n, "w", encoding="utf-8") as f:
            f.write(base_honey + "\n\n" + feedback)
        spec_n = f"{base}.repair{n}.{ext}"
        try:
            return run_specify_fn(
                honey_path=honey_n,
                codebase_root=codebase_root,
                output_path=spec_n,
                **specify_kwargs,
            )
        except Exception as e:  # a specify hiccup ends the loop honestly, never crashes it
            logger.error("repair: specify regenerate failed on attempt %d: %s", n, e)
            return None

    return regenerate
