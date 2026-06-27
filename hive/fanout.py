"""Fan-out stage — launches parallel copilot workers for each axis.

For each axis from the decompose output:
  1. Renders a comb prompt from the comb_contract_v2 template + axis brief
  2. Launches copilot subprocess in parallel
  3. Saves raw stdout to combs/comb_<axis_id>.txt
  4. Saves stderr to combs/err_<axis_id>.txt

Uses subprocess + concurrent.futures for parallel execution.
"""

import os
import subprocess
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from hive.parse import extract_first_json, is_comb_dict
from hive.providers import call_worker
from hive.retriever import FollowupNeed

logger = logging.getLogger("hive.fanout")

# Default comb contract template — loaded from file if available
DEFAULT_COMB_CONTRACT = """[Role] You are one Hivework free worker (drone). You dig into the single investigation axis assigned to you, and only that one. No code edits — investigation-only. Every claim MUST cite file:line evidence verified by actually opening the file with grep/read. No guessing.

[Target codebase root] {codebase_root} (git repo)

[Depth contract — no shallow combs] You MUST do the following:
1. **Execution reachability**: judge not that the code "exists" but whether it "actually runs." Check whether branch conditions, early returns, swallowed try/except, or **SQL WHERE gates** skip the block. Write "exists" and "reached" as distinct facts.
2. **Call-chain trace**: connect file:line with `→` from entry point → … → the DB write.
3. **Design contrast** (when possible): contrast the code's behavior against the spec intended by the design docs. A mismatch is the bug; a match is intended behavior — UNLESS the reporter declares that intended behavior itself wrong or unwanted, in which case the matching site is a DESIGN-CHANGE candidate (the site still must change), not a non-finding.
4. **blame** (if the axis is about regression/history): use `git log` / `git blame` to pin the introducing/modifying commit (hash + title) for the relevant lines. Also check "is it already fixed."
5. **Async state-overwrite races (reactive UI / shared state)**: when the symptom is "the value appears then disappears / flickers / is intermittently missing" AND one reactive state (a ref / store field / rendered badge or flag) is written by BOTH (a) a live event handler or optimistic local update, AND (b) an asynchronous fetch/refetch (silent SSE-driven reload, focus/visibility refresh, poll, re-open) that RESETS that same state to a default / null / empty value, then the prime root-cause candidate is the **later-resolving stale write clobbering the live value** — a write-write race — UNLESS a generation / version / sequence / timestamp guard provably discards the stale response. You MUST enumerate every writer site of that one piece of state (file:line) and state whether any such guard sits between them. Do NOT default to "the event was missed / the listener mounted late / the setter call is absent" when a second writer demonstrably overwrites an already-set value: "missed event" and "stale-overwrite" are DISTINCT mechanisms — *value set then cleared* ("appeared then vanished") points to the overwrite race, whereas *value never set* ("never appeared") points to the missed event. Pick the one the symptom and the writer-set actually support.

[Conclusion mandate — conclude, do NOT keep searching] Your job is to DELIVER A CONCLUSION, not to plan more searching. The moment you have opened the relevant files, STOP searching and synthesize what you found into `findings`. Do NOT emit your next search step — a tool-argument object such as {{"path":"...","pattern":"...","glob":"..."}} — or any prose as your answer. That is a search note, not a comb; it will be rejected and sent back to you. Even if your investigation genuinely turned up nothing, still CONCLUDE: return a well-formed comb with `findings`: [] and `termination` set. Decide with the evidence you already have.

[Output contract — comb] Output ONLY the single JSON object below. No prose, no text outside the JSON.
{{
  "axis_id": "<axis id>",
  "axis_title": "<axis title>",
  "trace": "<one-line call chain from entry point → … → DB/UI, with file:line. null if not applicable>",
  "findings": [
    {{
      "claim": "<the fact you verified>",
      "evidence": [{{"file":"<relative path>","lines":"<e.g. 49-78>","what":"<what those lines show>"}}],
      "reachable": "yes|no|conditional — does this code actually run in the target scenario, plus the condition",
      "confidence": "high|med|low"
    }}
  ],
  "design_ref": [{{"doc":"<design ID, e.g. M026 §8-1>","intended":"<what the design intended>","matches_code":"yes|no"}}],
  "regression": {{"commit":"<hash title / null>","what_changed":"<what changed and when / null>"}},
  "root_cause_signal": "<file:line if this axis directly pins the symptom's root cause, otherwise null>",
  "cross_refs": ["<other axis id>"],
  "termination": "resolved | needs_runtime | needs_external",
  "notes": "<one line. if unclosed, what else needs to be looked at>"
}}
"""


def load_comb_contract(contract_path: str | None, codebase_root: str) -> str:
    """Load the comb contract template.

    Args:
        contract_path: Path to comb_contract_v2.md, or None for default.
        codebase_root: Root path of target codebase (for template substitution).

    Returns:
        Comb contract template string.
    """
    if contract_path and os.path.exists(contract_path):
        with open(contract_path, 'r', encoding='utf-8') as f:
            template = f.read()
        # Contract files may contain JSON examples with literal braces, so avoid
        # str.format() here. Only the explicit placeholder is substituted.
        return template.replace("{codebase_root}", codebase_root)

    # Use default, substituting codebase_root
    return DEFAULT_COMB_CONTRACT.format(codebase_root=codebase_root)


def build_comb_prompt(contract: str, axis: dict[str, Any],
                      seed_text: str = "") -> str:
    """Build the full comb prompt for a single axis.

    Args:
        contract: The comb contract template.
        axis: Axis dict with id, title, brief.
        seed_text: Original seed text for context.

    Returns:
        Full prompt string.
    """
    axis_id = axis.get("id", axis.get("axis_id", "?"))
    title = axis.get("title", axis.get("axis_title", ""))
    brief = axis.get("brief", "")

    return f"""{contract}

[Assigned axis]
- axis_id: {axis_id}
- title: {title}
- brief: {brief}

[Original seed (full context)]
{seed_text}
"""


def is_comb_shaped(stdout: str) -> bool:
    """True iff ``stdout`` carries a comb-shaped JSON object — one with a
    ``findings`` list (the conclusion the drone was asked to produce).

    The dominant run-418/424 swarm failure was NOT an empty comb but a *non-empty
    non-comb* (CH hivework.default.0004.0008): the drone emitted its NEXT search as
    the answer — a tool-argument object like
    ``{"path":"","pattern":"create_button","glob":"*.vue"}`` — instead of
    synthesizing findings. That object decodes as valid JSON and is non-empty, so
    the loop's empty-comb guard (``not content.strip()``, http_tools) waves it
    through and it is scored as a 0-finding "success". A comb is a CONCLUSION; its
    signature is a ``findings`` list. A search-memo has no ``findings`` key, so this
    cleanly separates the two. Never raises: unparseable / non-object output (and a
    bare ``"ok"``) is simply not comb-shaped. Shape is decided by
    :func:`hive.parse.is_comb_dict` — the same predicate the pipeline-input gate
    uses, so telemetry and evidence agree on what counts as a comb."""
    try:
        obj = extract_first_json(stdout or "")
    except (ValueError, TypeError):
        return False
    return is_comb_dict(obj)


_RESPECIFY_BANNER = """[REJECTED — your previous output was not a comb]
Your last reply was NOT a valid investigation comb: it had no `findings` array. The \
dominant failure here is returning your NEXT search step — a tool-argument object \
such as {{"path": "...", "pattern": "...", "glob": "..."}} — or prose, as if it were \
the answer. That is a search note, not a conclusion.

Do NOT search further. CONCLUDE NOW from the evidence you have already gathered: emit \
the single comb JSON with a populated `findings` array (each finding citing \
file:line). If you genuinely found nothing, still return a well-formed comb with \
`findings`: [] and `termination` set — never a tool-argument object, never prose.

Your previous (rejected) output was:
{prev}

Now output ONLY the comb JSON, nothing else.

"""


def build_respecify_prompt(contract: str, axis: dict[str, Any], seed_text: str,
                           prev_output: str) -> str:
    """Build the re-specification prompt for a drone that returned a non-comb.

    Prepends an explicit rejection banner (echoing the offending output so the
    model sees its own mistake) to the original comb prompt, demanding a conclusion
    NOW rather than another search step. Pairs with the contract's
    ``[Conclusion mandate]`` — the static instruction plus this reactive rejection
    are the two halves CH 0004.0008 asked for."""
    prev = (prev_output or "").strip()[:800] or "(empty)"
    return _RESPECIFY_BANNER.format(prev=prev) + build_comb_prompt(contract, axis, seed_text)


_PROTECTED_AXIS_TERMS = (
    "wrt",
    "data-write",
    "data write",
    "write path",
    "data-mutation",
    "data mutation",
    "event-persistence",
    "event persistence",
    "event sink",
    "terminal event",
    "group event",
    "mutation sink",
    "persistence",
    "db write",
    "foreign key",
    "fk",
    "constraint",
)


_FK_PERSISTENCE_RE = re.compile(
    r"foreign\s*key|foreignkey|\bfk\b|integrity\s*error|integrityerror|"
    r"unique\s+constraint|not\s*null\s+constraint|check\s+constraint|"
    r"constraint\s+(?:failed|violat)|\bconstraint\b[^\n]{0,40}\bviolat|"
    r"\borphan(?:ed)?\b|\bcascade\b",
    re.IGNORECASE,
)
_WRITE_OR_FAIL_RE = re.compile(
    r"\binsert\b|\bupdate\b|\bdelete\b|\bcommit\b|\btransaction\b|\brollback\b|"
    r"\bwrite\b|\bpersist|\bmigrat|\b5\d\d\b|exception|error|fail|raise|traceback",
    re.IGNORECASE,
)
_MUTATION_VERB_RE = re.compile(
    r"\bdispose\b|\bdiscard(?:ed|ing|s)?\b|\bdelete\b|\bremov(?:e|ed|ing|al)\b|"
    r"\bdrop\b|\bclose\b|\bclosing\b|\binsert\b|\bupdate\b|\bsave\b|\bpersist|"
    r"\bcommit\b|\bwrite\b|\bmutat|"
    r"폐기|마감|삭제|제거|저장|기록|등록",
    re.IGNORECASE,
)
_SERVER_ERROR_RE = re.compile(
    r"\b5\d\d\b|internal\s+server\s+error|integrity\s*error|integrityerror|"
    r"\bexception\b|traceback|\braise[sd]?\b|\bfail(?:ed|s|ure)?\b|에러|오류|실패",
    re.IGNORECASE,
)
_BACKEND_FAILURE_RE = re.compile(
    r"\b5\d\d\b|internal\s+server\s+error|integrity\s*error|integrityerror|"
    r"\bexception\b|traceback|foreign\s*key|\bfk\b|constraint\s+(?:failed|violat)|"
    r"sqlite|sqlalchemy|database|db\s+error",
    re.IGNORECASE,
)
_FE_UI_SYMPTOM_RE = re.compile(
    r"\bfront[-\s]?end\b|\bui\b|\bux\b|\bbrowser\b|\bclient\b|"
    r"\bvue\b|\breact\b|\bcomponent\b|\bmodal\b|\bbutton\b|\bbadge\b|"
    r"\bheader\b|\brender(?:ed|ing)?\b|\bcop(?:y|ied|ies|ying)\b|"
    r"\bclipboard\b|\btoast\b|\btooltip\b|\bcss\b|\bclass(?:es)?\b|"
    r"프론트|클라이언트|브라우저|화면|표시|렌더|복사|클립보드|"
    r"뱃지|배지|헤더|버튼|모달|토스트|툴팁|스타일|색상",
    re.IGNORECASE,
)
_FE_AXIS_RE = re.compile(
    r"\bfront[-\s]?end\b|\bfrontend\b|client/src/|frontend/src/|web/src/|"
    r"ui/src/|\.vue\b|\.tsx\b|\.jsx\b|\.svelte\b",
    re.IGNORECASE,
)


def _axis_budget_text(axis: dict[str, Any]) -> str:
    sp = axis.get("search_plan") or {}
    bits = [
        str(axis.get(k, ""))
        for k in ("id", "axis_id", "title", "brief")
    ]
    bits.extend(str(v) for v in (sp.get("keywords") or []))
    bits.extend(str(v) for v in (sp.get("file_globs") or []))
    return " ".join(bits).replace("\\", "/").lower()


def _mutation_budget_seed(seed_text: str) -> bool:
    text = seed_text or ""
    if _FK_PERSISTENCE_RE.search(text) and _WRITE_OR_FAIL_RE.search(text):
        return True
    if _FE_UI_SYMPTOM_RE.search(text) and not _BACKEND_FAILURE_RE.search(text):
        return False
    return bool(_SERVER_ERROR_RE.search(text) and _MUTATION_VERB_RE.search(text))


def _fe_ui_budget_seed(seed_text: str) -> bool:
    return bool(_FE_UI_SYMPTOM_RE.search(seed_text or ""))


def _is_frontend_axis(axis: dict[str, Any]) -> bool:
    return bool(_FE_AXIS_RE.search(_axis_budget_text(axis)))


def _fe_seed_axis_score(axis: dict[str, Any], seed_text: str) -> int:
    seed = (seed_text or "").lower()
    text = _axis_budget_text(axis)
    score = 0
    if re.search(r"workflow|워크플로|결정|decision", seed, re.IGNORECASE):
        score += 2 * len(re.findall(r"workflow|decision|decided|status|side[-\s]?effect", text))
    if re.search(r"header|헤더", seed, re.IGNORECASE):
        score += 2 * len(re.findall(r"header|docheader", text))
    if re.search(r"badge|뱃지|배지|copied|복사됨", seed, re.IGNORECASE):
        score += 2 * len(re.findall(r"badge|copied|mentioncopy", text))
    if re.search(r"copy|clipboard|복사|멘트|mention", seed, re.IGNORECASE):
        score += len(re.findall(r"copy|clipboard|mention|fg:mention_copied", text))
    if re.search(r"timing|race|타이밍|사라지|안\s*보", seed, re.IGNORECASE):
        score += len(re.findall(r"race|timing|state|lifecycle|side[-\s]?effect|"
                                r"silent|refetch|timeout|clear|reset", text))
    if _is_frontend_axis(axis):
        score += 1
    if re.search(r"regression|blame|git log|design spec|ssot|documents/", text):
        score -= 30
    if re.search(r"hierarchy|data flow", text):
        score -= 8
    return score


def _is_protected_axis(axis: dict[str, Any], *, mutation_seed: bool = True) -> bool:
    """Return True for axes that should survive a tight fan-out call budget."""
    if not mutation_seed:
        return False
    text = _axis_budget_text(axis)
    return any(term in text for term in _PROTECTED_AXIS_TERMS)


def _apply_axis_call_budget(
    axes: list[dict[str, Any]],
    *,
    max_calls: int,
    respecify_retries: int,
    seed_text: str | None = None,
) -> list[dict[str, Any]]:
    """Trim axes for spend without letting symptom-irrelevant protected terms dominate."""
    if not (max_calls and max_calls > 0 and axes):
        return axes
    per_axis = 1 + max(0, respecify_retries)
    allowed = max(1, max_calls // per_axis)
    if len(axes) <= allowed:
        return axes

    # ``None`` preserves the historical helper default used by older direct tests.
    # Runtime fan-out passes the real seed, making write/FK protection symptom-aware.
    mutation_seed = True if seed_text is None else _mutation_budget_seed(seed_text)
    protected = [axis for axis in axes if _is_protected_axis(axis, mutation_seed=mutation_seed)]
    regular = [axis for axis in axes if not _is_protected_axis(axis, mutation_seed=mutation_seed)]
    reserved_fe = []
    if (seed_text is not None and not os.environ.get("HIVE_NO_FE_RESERVE")
            and _fe_ui_budget_seed(seed_text)):
        regular = sorted(
            regular,
            key=lambda axis: _fe_seed_axis_score(axis, seed_text),
            reverse=True,
        )
        reserved_fe = [axis for axis in regular if _is_frontend_axis(axis)][:1]

    ordered = protected + reserved_fe + regular if mutation_seed else reserved_fe + regular
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for axis in ordered:
        marker = id(axis)
        if marker in seen:
            continue
        selected.append(axis)
        seen.add(marker)
        if len(selected) >= allowed:
            break

    logger.warning("Fan-out: trimming %d axes to %d to honor max_calls=%d "
                   "(%d call(s)/axis; protected=%d; fe_reserved=%d; mutation_seed=%s)",
                   len(axes), allowed, max_calls, per_axis, len(protected),
                   len(reserved_fe), mutation_seed)
    return selected


def run_fanout(
    axes: list[dict[str, Any]],
    seed_text: str,
    codebase_root: str,
    workdir: str,
    contract_path: str | None = None,
    model: str = "gpt-5-mini",
    max_workers: int = 4,
    provider: str = "copilot",
    ledger=None,
    provider_kwargs: dict | None = None,
    respecify_retries: int = 1,
    max_calls: int = 0,
) -> dict[str, str]:
    """Run fan-out: launch parallel copilot workers for each axis.

    Args:
        axes: List of axis dicts from decompose output.
        seed_text: Original seed text.
        codebase_root: Target codebase root.
        workdir: Working directory for comb output files.
        contract_path: Path to comb_contract_v2.md.
        model: Model for copilot.
        max_workers: Max parallel workers.
        respecify_retries: How many extra respecify turns a non-comb reply gets before
            the axis gives up (config.fanout.retries). 1 preserves the single-pass
            behavior; a non-comb that resolves on the first respecify still stops there.
        max_calls: Hard ceiling on TOTAL drone calls this run (config.fanout.max_calls);
            0 = unlimited. When set, the axis list is trimmed pre-launch so the worst
            case ``axes x (1 + respecify_retries)`` stays at or under the ceiling.

    Returns:
        Dict mapping axis_id -> path to saved comb file.
    """
    combs_dir = os.path.join(workdir, "combs")
    os.makedirs(combs_dir, exist_ok=True)

    axes = _apply_axis_call_budget(axes, max_calls=max_calls,
                                   respecify_retries=respecify_retries,
                                   seed_text=seed_text)

    contract = load_comb_contract(contract_path, codebase_root)
    comb_files: dict[str, str] = {}

    def _run_one_axis(axis: dict[str, Any]) -> tuple[str, str, str, str, float, bool, str]:
        axis_id = axis.get("id", axis.get("axis_id", "?"))
        prompt = build_comb_prompt(contract, axis, seed_text)

        logger.info("  [fan-out] Launching worker for axis %s: %s",
                     axis_id, axis.get("title", "")[:60])

        comb_path = os.path.join(combs_dir, f"comb_{axis_id}.txt")
        err_path = os.path.join(combs_dir, f"err_{axis_id}.txt")

        def _one_call(stage_axis: str, the_prompt: str
                      ) -> tuple[str, float, bool, bool, str]:
            """Run ONE billed worker call + its ledger row. Returns
            ``(stdout, latency_s, exit_ok, shaped, err)``. Never raises (a timeout
            degrades to a failed row + TIMEOUT marker), so the caller can retry or
            fall through cleanly.

            Begin the ledger row BEFORE the (up to 600s) call so the in-flight worker
            is visible and a timeout still leaves a 'failed' row. begin/finish_call are
            lock-guarded, so the parallel pool can record safely. Ledger honesty
            (CH 0004.0008): a non-empty reply that is not comb-shaped — a search-memo —
            is recorded ok=0 even on exit 0, so comb-yield telemetry stops counting
            noise as success."""
            cid = ledger.begin_call("fanout", stage_axis, provider, model, the_prompt,
                                    comb_path) if ledger is not None else None
            try:
                result = call_worker(provider, model, the_prompt, cwd=codebase_root,
                                     timeout=600,
                                     on_start=(lambda: ledger.mark_running(cid))
                                     if (ledger is not None and cid is not None) else None,
                                     **(provider_kwargs or {}))
            except subprocess.TimeoutExpired:
                logger.error("  [fan-out] Axis %s TIMED OUT", stage_axis)
                if ledger is not None:
                    ledger.finish_call(cid, output="", latency_s=600.0, ok=False,
                                       err="TIMEOUT")
                return (f"TIMEOUT: worker for axis {axis_id} exceeded 600s limit",
                        600.0, False, False, "TIMEOUT")
            exit_ok = result.exit_code == 0
            shaped = is_comb_shaped(result.stdout)
            if not exit_ok:
                err = result.stderr[:200]
            elif not shaped:
                err = "non-comb output (no findings array)"
            else:
                err = ""
            if ledger is not None:
                ledger.finish_call(cid, output=result.stdout, latency_s=result.latency_s,
                                   ok=exit_ok and shaped, err=err,
                                   real_tokens=result.real_tokens)
            return result.stdout, result.latency_s, exit_ok, shaped, err

        stdout, latency_s, exit_ok, shaped, err = _one_call(axis_id, prompt)
        used_prompt = prompt

        # Shape-reject retry (CH 0004.0008 fix ①): a call that completed but produced a
        # non-comb — the "next search" tool-arg object that dominated run 424 — gets ONE
        # re-specification turn that rejects the memo and demands a conclusion NOW. This
        # is the retry the empty-comb guard never fired (its trigger, empty content, was
        # never met by non-empty noise); paired with the contract's [Conclusion mandate]
        # (fix ②). An empty reply is also non-comb, so this subsumes the empty case with
        # a stronger instruction. Skipped on TIMEOUT (exit_ok False) — nothing to respecify.
        attempt = 0
        while exit_ok and not shaped and attempt < max(0, respecify_retries):
            attempt += 1
            label = "re-specifying once" if respecify_retries == 1 else \
                f"re-specifying (attempt {attempt}/{respecify_retries})"
            logger.warning("  [fan-out] Axis %s returned a non-comb (search-memo?); %s",
                           axis_id, label)
            respecify = build_respecify_prompt(contract, axis, seed_text, stdout)
            suffix = "#respecify" if respecify_retries == 1 else f"#respecify{attempt}"
            r_out, r_lat, r_exit_ok, r_shaped, r_err = _one_call(f"{axis_id}{suffix}",
                                                                 respecify)
            # Adopt the retry when it is a real comb, or when it salvages an empty first
            # reply; otherwise keep the first output (neither is a comb, but the first at
            # least carries whatever the drone produced).
            if r_shaped or not stdout.strip():
                stdout, latency_s, exit_ok, shaped, err = r_out, r_lat, r_exit_ok, r_shaped, r_err
                used_prompt = respecify
            if r_shaped:
                break  # a real comb ends the retry budget early

        # G8-race guard: combs_dir is created once before the pool launches, but
        # call_worker above can run for minutes. If anything external removes the
        # dir in that window (a concurrent run sharing the default workdir, tmp
        # cleanup), the write below dies with FileNotFoundError and aborts the whole
        # pipeline. Re-ensure the parent exists right before writing.
        os.makedirs(combs_dir, exist_ok=True)
        with open(comb_path, 'w', encoding='utf-8') as f:
            f.write(stdout)
        with open(err_path, 'w', encoding='utf-8') as f:
            f.write(err)

        logger.info("  [fan-out] Axis %s done (exit_ok=%s, comb=%s, stdout=%d bytes)",
                    axis_id, exit_ok, shaped, len(stdout))
        return axis_id, comb_path, used_prompt, stdout, latency_s, exit_ok, err

    # Launch in parallel
    logger.info("Fan-out: launching %d workers (max_parallel=%d)",
                len(axes), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one_axis, axis): axis for axis in axes}
        for future in as_completed(futures):
            axis_id, comb_path, *_ = future.result()
            comb_files[axis_id] = comb_path

    # Each swarm call is recorded inside _run_one_axis (begin before the call,
    # finish on success/timeout) so in-flight workers are visible live.
    logger.info("Fan-out complete: %d combs saved", len(comb_files))
    return comb_files


# ── M013 B3: targeted, capped REINFORCEMENT (scouts, NOT a swarm) ──────────────
# Distinct from the open-ended fan-out above (a real swarm): reinforcement fires ONLY on
# an axis the queen flagged (coverage_risk) AND whose blind local FIND returned nothing
# (needs_reinforcement, B2). A FEW quality SCOUT agents (roles.scout, 120b) dig for the
# evidence the blind grep missed; we lift the file:lines they cite and hand them back as a
# FollowupNeed the caller windows LOCALLY (free) into the bundle before judge. This is the
# opposite of a swarm: more scouts on one thin axis just re-find the same files (the swarm
# pattern — many cheap independent agents averaged — lives in judge's best-of-N vote, not
# here). Gated by cfg.reinforce.enabled (default off) and bounded by a shared run budget.


class ReinforceBudget:
    """Thread-safe ceiling on total reinforcement (scout) calls per investigate run.

    Scouts run inside a thread pool, so the per-run ``max_total_calls`` cap is enforced
    with a lock: each axis ``take(n)``s as many scout slots as remain (≤ its own
    ``max_workers`` request). Once drained, further axes get 0 and skip reinforcement —
    the hard one-number budget shape mirrors ``JudgeConfig.max_total_calls``.
    """

    def __init__(self, total: int):
        self._remaining = max(0, int(total))
        self._lock = threading.Lock()

    def take(self, want: int) -> int:
        want = max(0, int(want))
        with self._lock:
            granted = min(want, self._remaining)
            self._remaining -= granted
            return granted

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._remaining


def _extract_comb_evidence(comb_stdout: str) -> tuple[list[str], list[str]]:
    """Pull cited (file paths, root-cause symbols) out of one comb JSON. Never raises.

    A scout's value on a thin axis is the FILES it located that the blind grep missed;
    we feed those back as a re-search scope. ``root_cause_signal`` (``file:line``) and
    ``findings[].evidence[].file`` are the cited paths; the basename (sans extension) of a
    root-cause file is offered as a symbol seed. A malformed comb yields ([], [])."""
    try:
        obj = extract_first_json(comb_stdout or "")
    except (ValueError, TypeError):
        return [], []
    if not isinstance(obj, dict):
        return [], []
    files: list[str] = []
    symbols: list[str] = []

    def _add_file(f: Any) -> None:
        f = str(f or "").strip()
        if f and f not in files:
            files.append(f)

    sig = str(obj.get("root_cause_signal") or "").strip()
    if sig:
        _add_file(sig.split(":", 1)[0])
    for finding in obj.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        for ev in finding.get("evidence") or []:
            if isinstance(ev, dict):
                _add_file(ev.get("file"))
    return files, symbols


def reinforce_thin_axis(task: dict[str, Any], sp, seed_text: str, code_root: str, *,
                        cfg, budget: "ReinforceBudget | None",
                        ledger=None, provider_kwargs: dict | None = None):
    """Capped scout reinforcement for ONE ``needs_reinforcement`` axis (B3).

    Returns a :class:`hive.retriever.FollowupNeed` (scope = the files the scouts cited,
    grepped by the axis's own keywords) for the caller to window LOCALLY into the bundle
    before judge — or ``None`` when reinforcement is disabled, out of budget, or the
    scouts cited nothing. Never raises: a flaky/empty scout degrades to no reinforcement,
    so an enabled-but-unlucky run is no worse than the un-reinforced path.
    """
    if not getattr(cfg.reinforce, "enabled", False):
        return None
    axis_id = sp.axis_id
    want = max(1, int(cfg.reinforce.max_workers))
    granted = budget.take(want) if budget is not None else 0
    if granted <= 0:
        logger.info("   reinforce[%s]: skipped (budget exhausted)", axis_id)
        return None

    role = cfg.role("scout")
    pk = dict(provider_kwargs or {})
    prompt = build_comb_prompt(load_comb_contract(None, code_root), task, seed_text)
    logger.info("   reinforce[%s]: %d scout(s) (%s/%s) on thin axis",
                axis_id, granted, role.provider, role.model)

    def _one(i: int) -> str:
        call_id = ledger.begin_call("scout", f"{axis_id}#reinforce{i}", role.provider,
                                    role.model, prompt) if ledger is not None else None
        try:
            wr = call_worker(role.provider, role.model, prompt, cwd=code_root,
                             timeout=600,
                             on_start=(lambda: ledger.mark_running(call_id))
                             if (ledger is not None and call_id is not None) else None,
                             **pk)
        except subprocess.SubprocessError as e:
            if ledger is not None:
                ledger.finish_call(call_id, output="", latency_s=0.0, ok=False,
                                   err=str(e)[:200])
            return ""
        if ledger is not None:
            ledger.finish_call(call_id, output=wr.stdout,
                               latency_s=wr.latency_s, ok=wr.exit_code == 0,
                               err=wr.stderr[:200] if wr.exit_code != 0 else "",
                               real_tokens=wr.real_tokens)
        return wr.stdout if wr.exit_code == 0 else ""

    if granted == 1:
        outs = [_one(0)]
    else:
        with ThreadPoolExecutor(max_workers=granted) as pool:
            outs = list(pool.map(_one, range(granted)))

    files: list[str] = []
    for out in outs:
        cited, _syms = _extract_comb_evidence(out)
        for f in cited:
            if f not in files:
                files.append(f)
    if not files:
        logger.info("   reinforce[%s]: scouts cited no new files — no enrichment", axis_id)
        return None
    logger.info("   reinforce[%s]: scouts cited %d file(s) → local re-window", axis_id,
                len(files))
    # Scope the local re-search to the cited files, grepped by the axis's own keywords.
    return FollowupNeed(axis_id=axis_id, symbols=[], greps=list(sp.keywords),
                        file_globs=files)
