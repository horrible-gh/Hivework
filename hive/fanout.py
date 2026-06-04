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

from hive.parse import extract_first_json
from hive.providers import call_worker
from hive.retriever import FollowupNeed

logger = logging.getLogger("hive.fanout")

# Default comb contract template — loaded from file if available
DEFAULT_COMB_CONTRACT = """[Role] You are one Hivework free worker (drone). You dig into the single investigation axis assigned to you, and only that one. No code edits — investigation-only. Every claim MUST cite file:line evidence verified by actually opening the file with grep/read. No guessing.

[Target codebase root] {codebase_root} (git repo)

[Depth contract — no shallow combs] You MUST do the following:
1. **Execution reachability**: judge not that the code "exists" but whether it "actually runs." Check whether branch conditions, early returns, swallowed try/except, or **SQL WHERE gates** skip the block. Write "exists" and "reached" as distinct facts.
2. **Call-chain trace**: connect file:line with `→` from entry point → … → the DB write.
3. **Design contrast** (when possible): contrast the code's behavior against the spec intended by the design docs. A mismatch is the bug; a match is intended behavior.
4. **blame** (if the axis is about regression/history): use `git log` / `git blame` to pin the introducing/modifying commit (hash + title) for the relevant lines. Also check "is it already fixed."

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

    Returns:
        Dict mapping axis_id -> path to saved comb file.
    """
    combs_dir = os.path.join(workdir, "combs")
    os.makedirs(combs_dir, exist_ok=True)

    contract = load_comb_contract(contract_path, codebase_root)
    comb_files: dict[str, str] = {}

    def _run_one_axis(axis: dict[str, Any]) -> tuple[str, str, str, str, float, bool, str]:
        axis_id = axis.get("id", axis.get("axis_id", "?"))
        prompt = build_comb_prompt(contract, axis, seed_text)

        logger.info("  [fan-out] Launching worker for axis %s: %s",
                     axis_id, axis.get("title", "")[:60])

        comb_path = os.path.join(combs_dir, f"comb_{axis_id}.txt")
        err_path = os.path.join(combs_dir, f"err_{axis_id}.txt")

        try:
            result = call_worker(provider, model, prompt, cwd=codebase_root, timeout=600,
                                 **(provider_kwargs or {}))
            err_msg = result.stderr[:200] if result.exit_code != 0 else ""

            # G8-race guard: combs_dir is created once before the pool launches, but
            # call_worker above can run for minutes. If anything external removes the
            # dir in that window (a concurrent run sharing the default workdir, tmp
            # cleanup), the write below dies with FileNotFoundError and aborts the whole
            # pipeline. Re-ensure the parent exists right before writing.
            os.makedirs(combs_dir, exist_ok=True)
            with open(comb_path, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            with open(err_path, 'w', encoding='utf-8') as f:
                f.write(result.stderr)

            logger.info("  [fan-out] Axis %s done (exit=%d, stdout=%d bytes)",
                        axis_id, result.exit_code, len(result.stdout))
            return axis_id, comb_path, prompt, result.stdout, result.latency_s, result.exit_code == 0, err_msg

        except subprocess.TimeoutExpired:
            logger.error("  [fan-out] Axis %s TIMED OUT", axis_id)
            timeout_msg = f"TIMEOUT: worker for axis {axis_id} exceeded 600s limit"
            os.makedirs(combs_dir, exist_ok=True)  # same G8-race guard (600s window)
            with open(comb_path, 'w', encoding='utf-8') as f:
                f.write(timeout_msg)
            with open(err_path, 'w', encoding='utf-8') as f:
                f.write("TIMEOUT")
            return axis_id, comb_path, prompt, timeout_msg, 600.0, False, "TIMEOUT"

    # Launch in parallel
    logger.info("Fan-out: launching %d workers (max_parallel=%d)",
                len(axes), max_workers)

    results: list[tuple[str, str, str, str, float, bool, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one_axis, axis): axis for axis in axes}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            axis_id, comb_path, *_ = result
            comb_files[axis_id] = comb_path

    if ledger is not None:
        for axis_id, comb_path, prompt, output, latency_s, ok, err_msg in results:
            ledger.record_call("swarm", axis_id, provider, model,
                               prompt=prompt, output=output, latency_s=latency_s,
                               comb_path=comb_path, ok=ok, err=err_msg)

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
        try:
            wr = call_worker(role.provider, role.model, prompt, cwd=code_root,
                             timeout=600, **pk)
        except subprocess.SubprocessError:
            return ""
        if ledger is not None:
            ledger.record_call("scout", f"{axis_id}#reinforce{i}", role.provider,
                               role.model, prompt=prompt, output=wr.stdout,
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
