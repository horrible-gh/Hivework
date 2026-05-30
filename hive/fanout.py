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
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

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
  "termination": "resolved | needs_runtime | needs_external | needs_pm",
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

    def _run_one_axis(axis: dict[str, Any]) -> tuple[str, str]:
        axis_id = axis.get("id", axis.get("axis_id", "?"))
        prompt = build_comb_prompt(contract, axis, seed_text)

        logger.info("  [fan-out] Launching worker for axis %s: %s",
                     axis_id, axis.get("title", "")[:60])

        comb_path = os.path.join(combs_dir, f"comb_{axis_id}.txt")
        err_path = os.path.join(combs_dir, f"err_{axis_id}.txt")

        try:
            result = subprocess.run(
                [shutil.which("copilot.cmd") or shutil.which("copilot") or "copilot", "--allow-all", "--model", model],
                input=prompt,  # prompt via stdin (cmd.exe argv truncates long/Korean prompts)
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=codebase_root,
                timeout=600,  # 10 min per axis
            )

            with open(comb_path, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            with open(err_path, 'w', encoding='utf-8') as f:
                f.write(result.stderr)

            logger.info("  [fan-out] Axis %s done (exit=%d, stdout=%d bytes)",
                        axis_id, result.returncode, len(result.stdout))

        except subprocess.TimeoutExpired:
            logger.error("  [fan-out] Axis %s TIMED OUT", axis_id)
            with open(comb_path, 'w', encoding='utf-8') as f:
                f.write(f"TIMEOUT: worker for axis {axis_id} exceeded 600s limit")
            with open(err_path, 'w', encoding='utf-8') as f:
                f.write("TIMEOUT")

        return axis_id, comb_path

    # Launch in parallel
    logger.info("Fan-out: launching %d workers (max_parallel=%d)",
                len(axes), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one_axis, axis): axis for axis in axes}
        for future in as_completed(futures):
            axis_id, comb_path = future.result()
            comb_files[axis_id] = comb_path

    logger.info("Fan-out complete: %d combs saved", len(comb_files))
    return comb_files
