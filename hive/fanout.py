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
        # The contract file uses the codebase root directly — return as-is
        return template

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

    # Pre-launch budget ceiling: trim axes so worst-case calls <= max_calls. Deterministic
    # and opt-in (0 = no cap), so today's uncapped behavior is unchanged unless configured.
    if max_calls and max_calls > 0 and axes:
        per_axis = 1 + max(0, respecify_retries)
        allowed = max(1, max_calls // per_axis)
        if len(axes) > allowed:
            logger.warning("Fan-out: trimming %d axes to %d to honor max_calls=%d "
                           "(%d call(s)/axis)", len(axes), allowed, max_calls, per_axis)
            axes = axes[:allowed]

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
            cid = ledger.begin_call("swarm", stage_axis, provider, model, the_prompt,
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
