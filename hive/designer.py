"""Designer (설계자) stage — authors the acceptance-criteria design that arms the keymaster.

Why this module exists
----------------------
Before group 0079 the hive had an investigator (finds the cause + location) and a
mutator (specify → apply), but no seat that turned "what does DONE look like?" into a
document the keymaster (``hive.acceptance_synth``, box-0) could grind into a failing
red test. A human had to author the ``## 수용기준`` design by hand, so an unmanned seed
either shipped unarmed (the keymaster silently no-op'd) or stopped at a human boundary.

The designer fills that seat: it reads the seed, the GROUNDED investigate verdicts
(file:line + reason) and the local honey, then authors a design whose ``## 수용기준``
YAML carries EXPLICIT ``oracle:`` blocks — the only shape that reaches the keymaster's
certified L2~L7b range (prose derivation caps at a single route + single field, NR0003
§3.2). It PRECHECKS every criterion against the SAME function the keymaster runs at
synthesis (``acceptance_synth.detect_acceptance``) so a passing precheck structurally
guarantees a passing armament — no duplicated check logic, one source of truth.

Discipline (L0005 §4 invariant)
------------------------------
On the designer path "proceed with NO acceptance criteria, silently" NEVER happens.
A grounded seed whose criteria all decline returns ``no_go`` (the caller halts before
specify — ``design_no_go`` — rather than arming-less progress). The legacy fail-open
behaviour (``_synthesize_acceptance_red_test`` skips a declined criterion) survives ONLY
on the no-designer path (kill switch ``HIVE_NO_DESIGNER`` or a seed with no grounded
verdict, where the designer abstains). This module is pure-local except the single
author model call, never raises to the caller (a model failure → ``no_go``), and grounds
nothing it did not read.

Logic SSOT: hivework.default.0079.0005-L (design: 0004-D, investigation: 0003-NR).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from hive.acceptance_synth import (
    ACCEPTANCE_MARKER,
    detect_acceptance,
    read_acceptance_criteria,
)

logger = logging.getLogger("hive.designer")

# ── L0005 §1 parameters ──────────────────────────────────────────────────────
KILL_SWITCH_ENV = "HIVE_NO_DESIGNER"     # set → bypass the whole stage (legacy fallback)
MIN_GROUNDED_VERDICTS = 1                # grounded verdicts needed to author at all
MAX_AUTHORING_ATTEMPTS = 2              # first author + one re-author on decline
MIN_VALID_CRITERIA = 1                  # prechecked-valid criteria needed to proceed
MAX_CRITERIA = 5                        # criteria per design (cost / legibility cap)

# recipe judgement the author folds into the design (L0005 §2.4). Parsed loosely from a
# ``recipe: code_bug`` / ``recipe = code_feature`` line so the author can place it anywhere.
_RECIPE_RE = re.compile(r"recipe\s*[:=]\s*`?(code_bug|code_feature)`?", re.I)

# yaml is used only to re-emit the ## 수용기준 block when demoting a partially-declined
# design. Guarded like acceptance_synth: without it read_acceptance_criteria already
# returns [] (so we never reach demote), and demote falls back to leaving the text as-is.
try:  # pragma: no cover - import guard
    import yaml as _yaml
except Exception:  # pragma: no cover
    _yaml = None


# ── author prompt ─────────────────────────────────────────────────────────────
DESIGNER_SYSTEM = """# ROLE: DESIGNER (설계자) — acceptance-criteria author, automated pipeline

You are the DESIGNER in an unmanned Hivework pipeline. The investigator has already
grounded WHERE the change lives. Your ONLY job is to write a design whose acceptance
criteria (``## 수용기준``) the keymaster can grind into an executable red→green test —
so ``apply --verify`` certifies the work BY EXECUTION, not by a human reading prose.

## Output format (STRICT)

Emit MARKDOWN only. It MUST contain a ``## 수용기준`` heading followed by a YAML list.
Each list item is one criterion:

```
## 수용기준
- id: AC1
  prose: <one plain sentence a human can read>
  oracle:
    kind: http_read            # or unit_value
    verb: get
    route: /api/v1/projects    # a route the investigation actually grounded
    json_path: modules         # a response field the code actually returns
    must: non_empty            # exists | non_empty | equals (equals needs `expected`)
```

Also emit, on its own line anywhere, a recipe judgement:  ``recipe: code_bug``  (or
``recipe: code_feature``) — code_feature only when the seed asks for NEW behaviour.

## Oracle vocabulary (do NOT invent words — this is the keymaster's exact contract)

- kind: http_read | unit_value
- must: exists | non_empty | equals | equals_len | equals_path | increases_by | decreases_by | delta_equals
- structure axis — choose ONE per criterion:
  - single route + json_path (DEFAULT): a lone `route` + `json_path` + `must`
  - asserts: [ {json_path, must, expected} ... ]      # ≥2 fields on ONE response
  - steps: [ {verb: post, route, body, expect_status}, {verb: get, route, asserts} ]  # mutate then read back
  - reads: [ {verb, route, asserts} ... ]             # ≥2 independent routes
  - delta: {observe:{verb,route,json_path}, mutate:{verb,route,body,expect_status}, must: increases_by, by: 1}
- decision tree: change-magnitude→delta ; mutate-then-confirm→steps ; ≥2 routes→reads ;
  ≥2 fields on one response→asserts ; else→single route+json_path.

## Grounding discipline (추측 금지 — this is the whole point)

- Use ONLY routes, json_paths, targets and literal expected values that the grounded
  verdicts or the honey ACTUALLY confirm. Never guess a route, a field, or a value.
- If you cannot confirm an exact expected value, use ``must: exists`` or ``must: non_empty``
  — do NOT write ``equals`` against a guessed literal (the keymaster will decline it).
- unit_value target is ``file.py::symbol`` (only when you saw that symbol grounded).
- Write ≤ %(max_criteria)d criteria. Fewer, certain criteria beat many shaky ones.
""" % {"max_criteria": MAX_CRITERIA}


def _grounded(verdicts: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """The verdicts whose judge located a concrete file:line (``verdict.located``)."""
    return [v for v in (verdicts or [])
            if isinstance(v, dict) and (v.get("verdict") or {}).get("located")]


def _format_grounding(grounded: list[dict[str, Any]]) -> str:
    """One evidence line per grounded verdict: axis → file:line — reason (+ design_change tag)."""
    lines: list[str] = []
    for v in grounded:
        vd = v.get("verdict") or {}
        loc = f"{vd.get('file', '')}:{vd.get('lines', '')}".strip(":")
        tag = ""
        if str(vd.get("type", "")).strip().lower() == "design_change":
            tag = "  [DESIGN-CHANGE: code matches its own spec but the requested result is wrong]"
        lines.append(f"- axis {v.get('axis_id', '?')} — {v.get('title', '')}: "
                     f"{loc} — {vd.get('reason', '')}{tag}")
    return "\n".join(lines)


def _format_declines(declines: list[dict[str, Any]]) -> str:
    """Re-author feedback: each previously-declined criterion + why, so the author fixes it."""
    if not declines:
        return ""
    lines = ["", "## Your PREVIOUS attempt's criteria were DECLINED by the keymaster precheck.",
             "Fix these — a decline means the oracle did not resolve against live code "
             "(ambiguous route, missing field, an `equals` with no grounded literal, or "
             "an invented oracle word). Re-ground each against the evidence above:"]
    for d in declines:
        lines.append(f"- {d.get('id', '?')} [{d.get('reason', '')}]: {d.get('prose', '')}")
    return "\n".join(lines)


def build_author_prompt(seed_text: str, grounded: list[dict[str, Any]],
                        honey_text: str, last_declines: list[dict[str, Any]]) -> str:
    """Assemble the designer's author prompt (system + seed + grounding + honey + retry feedback)."""
    return f"""{DESIGNER_SYSTEM}

## Requested change (seed)
{(seed_text or '').strip()}

## Grounded localisations (investigation evidence — REAL routes/fields/symbols live here)
{_format_grounding(grounded) or '(none)'}

## Investigation honey (supporting context — verify, do not trust blindly)
{(honey_text or '').strip() or '(none)'}
{_format_declines(last_declines)}
"""


# ── recipe parsing ─────────────────────────────────────────────────────────────
def parse_recipe(design_text: str) -> str | None:
    """The author's ``recipe: code_bug|code_feature`` judgement, or ``None`` when absent.

    None means "the author did not judge" → the caller falls back to the regex classifier
    (``select_recipe``), preserving the operator > designer > classifier priority (L §2.4)."""
    m = _RECIPE_RE.search(design_text or "")
    return m.group(1).lower() if m else None


# ── precheck (L §2.3) ───────────────────────────────────────────────────────────
def precheck_criteria(criteria: list[dict[str, Any]],
                      codebase_root: str) -> dict[str, list]:
    """Partition criteria into keymaster-acceptable (``valid``) and declined.

    Calls the EXACT function the keymaster runs at synthesis time
    (``acceptance_synth.detect_acceptance``): a non-None symptom means the oracle grounds
    and will arm; None is a decline. No duplicated check logic — a precheck pass
    structurally guarantees a synth-time pass (L §2.3 single-truth). Observation-only:
    writes nothing, never raises (a detect exception is treated as a decline)."""
    valid: list[dict[str, Any]] = []
    declined: list[dict[str, Any]] = []
    for ac in criteria:
        try:
            symptom = detect_acceptance(ac, codebase_root)
        except Exception:  # a grounding fault is a decline, never a crash
            symptom = None
        if symptom is None:
            declined.append({"id": str(ac.get("id", "")), "reason": "keymaster_decline",
                             "prose": str(ac.get("prose", ""))})
        else:
            valid.append(ac)
    return {"valid": valid, "declined": declined}


# ── demote (L §2.1 demote_declined) ─────────────────────────────────────────────
def _dump_criteria_section(valid: list[dict[str, Any]]) -> str:
    """Re-emit the ``## 수용기준`` block holding ONLY the valid criteria."""
    body = _yaml.safe_dump(valid, allow_unicode=True, sort_keys=False) if _yaml else ""
    return f"{ACCEPTANCE_MARKER}\n\n{body}\n"


def _replace_marker_section(design_text: str, new_section: str) -> str:
    """Swap the ``## 수용기준`` block (marker → next heading) for ``new_section``."""
    if ACCEPTANCE_MARKER not in design_text:
        return design_text.rstrip() + "\n\n" + new_section
    idx = design_text.index(ACCEPTANCE_MARKER)
    before = design_text[:idx]
    after = design_text[idx + len(ACCEPTANCE_MARKER):]
    nxt = re.search(r"\n#{1,2}\s", after)
    rest = after[nxt.start():].lstrip("\n") if nxt else ""
    return before + new_section + (rest + "\n" if rest else "")


def demote_declined(design_text: str, valid: list[dict[str, Any]],
                    declined: list[dict[str, Any]]) -> str:
    """Proceed with a partially-declined design: strip declined criteria from the keymaster
    input and preserve them as reference prose (L §2.1 — excluded from arming, info kept)."""
    if not declined or _yaml is None:
        return design_text
    out = _replace_marker_section(design_text, _dump_criteria_section(valid))
    appendix = ["", "## 참고: 강등된 수용기준 (키마스터 입력 제외 — 정보 보존)", ""]
    for d in declined:
        appendix.append(f"- {d.get('id', '?')} [{d.get('reason', '')}]: {d.get('prose', '')}")
    return out.rstrip() + "\n" + "\n".join(appendix) + "\n"


# ── author model call ───────────────────────────────────────────────────────────
def _strip_tool_traces(raw: str) -> str:
    """Drop leading ● / ✗ / │ / └ tool-trace lines a CLI worker may prefix (assemble parity)."""
    lines = (raw or "").split("\n")
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(("●", "✗", "│", "└")):
            start = i + 1
            continue
        if not s and start == i:
            start = i + 1
            continue
        break
    return "\n".join(lines[start:])


def _make_author(role, codebase_root: str, provider_kwargs: dict | None,
                 ledger) -> Callable[[str], str]:
    """A one-shot author callable bound to the designer role, ledgered like every stage."""
    from hive.providers import call_worker

    def _author(prompt: str) -> str:
        call_id = (ledger.begin_call("designer", "designer", role.provider, role.model,
                                     prompt) if ledger is not None else None)
        try:
            wr = call_worker(
                role.provider, role.model, prompt, cwd=codebase_root,
                timeout=(role.timeout_sec or 600),
                on_start=((lambda: ledger.mark_running(call_id))
                          if (ledger is not None and call_id is not None) else None),
                **(provider_kwargs or {}))
        except Exception as e:
            if ledger is not None:
                ledger.finish_call(call_id, output="", latency_s=0.0, ok=False,
                                   err=str(e)[:200])
            raise
        if ledger is not None:
            ledger.finish_call(call_id, output=wr.stdout, latency_s=wr.latency_s,
                               ok=wr.exit_code == 0,
                               err=wr.stderr[:200] if wr.exit_code != 0 else "",
                               real_tokens=wr.real_tokens)
        return _strip_tool_traces(wr.stdout)

    return _author


# ── main loop (L §2.1) ──────────────────────────────────────────────────────────
def run_designer(seed_text: str, verdicts: list[dict[str, Any]] | None,
                 honey_text: str, codebase_root: str, role, *,
                 provider_kwargs: dict | None = None, ledger=None,
                 author_fn: Callable[[str], str] | None = None,
                 min_grounded: int = MIN_GROUNDED_VERDICTS,
                 max_attempts: int = MAX_AUTHORING_ATTEMPTS,
                 min_valid: int = MIN_VALID_CRITERIA,
                 max_criteria: int = MAX_CRITERIA) -> dict[str, Any]:
    """Author + precheck an acceptance design; decide proceed / no_go / needs_reinvestigation.

    Returns one of (L §2.1 / §3):
    - ``{decision: "proceed", design_text, valid_ids, declined, recipe, attempts, truncated}``
    - ``{decision: "no_go", reason: "all_criteria_declined" | "model_error", ...}``
    - ``{decision: "needs_reinvestigation", reason: "no_grounded_verdict", attempts: 0}``

    ``author_fn`` (prompt → design text) is injectable for tests; the default makes the
    single designer model call. Never raises — a model failure becomes ``no_go``."""
    grounded = _grounded(verdicts)
    if len(grounded) < min_grounded:
        logger.info("designer: %d grounded verdict(s) < %d — needs_reinvestigation",
                    len(grounded), min_grounded)
        return {"decision": "needs_reinvestigation", "reason": "no_grounded_verdict",
                "attempts": 0}

    author = author_fn or _make_author(role, codebase_root, provider_kwargs, ledger)
    last_declines: list[dict[str, Any]] = []
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            design_text = author(build_author_prompt(
                seed_text, grounded, honey_text, last_declines))
        except Exception as e:
            logger.error("designer: author call failed on attempt %d: %s", attempt, e)
            return {"decision": "no_go", "reason": "model_error",
                    "error": str(e)[:200], "attempts": attempt}

        criteria = read_acceptance_criteria(design_text)
        truncated = False
        if len(criteria) > max_criteria:
            logger.warning("designer: %d criteria > cap %d — prechecking the first %d "
                           "(the rest are dropped, recorded)",
                           len(criteria), max_criteria, max_criteria)
            criteria = criteria[:max_criteria]
            truncated = True

        result = precheck_criteria(criteria, codebase_root)
        if len(result["valid"]) >= min_valid:
            design_out = demote_declined(design_text, result["valid"], result["declined"])
            valid_ids = [str(c.get("id", "")) for c in result["valid"]]
            logger.info("designer: proceed on attempt %d — %d valid criterion(s) [%s]"
                        "%s", attempt, len(valid_ids), ", ".join(valid_ids),
                        f", {len(result['declined'])} demoted" if result["declined"] else "")
            return {"decision": "proceed", "design_text": design_out,
                    "valid_ids": valid_ids, "declined": result["declined"],
                    "recipe": parse_recipe(design_text), "attempts": attempt,
                    "truncated": truncated}

        last_declines = result["declined"]
        logger.info("designer: attempt %d produced 0 valid criteria (%d declined) — %s",
                    attempt, len(result["declined"]),
                    "re-authoring with decline feedback" if attempt < max_attempts
                    else "exhausted → no_go")

    return {"decision": "no_go", "reason": "all_criteria_declined",
            "declines": last_declines, "attempts": attempt}
