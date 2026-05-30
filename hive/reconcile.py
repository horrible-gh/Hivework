"""Reconcile loop — re-investigates conflicts until convergence.

When conflict-scan detects conflicts:
  1. Generates a narrowed re-investigation brief (citing both sides' file:line evidence)
  2. Launches a copilot worker for re-investigation
  3. Parses the new comb
  4. Re-runs conflict-scan
  5. Repeats until convergence or round cap (recipe default: 2)

The reconcile brief MUST:
  - Quote both conflicting conclusions with their file:line evidence
  - Ask a narrowed question focused on resolving the specific contradiction
  - Always ask about execution reachability ("does that code actually run?")
"""

import os
import subprocess
import shutil
import logging
from typing import Any

from hive.parse import extract_first_json
from hive.conflict_scan import scan_conflicts

logger = logging.getLogger("hive.reconcile")


def build_reconcile_brief(conflicts: list[dict[str, Any]],
                          combs: list[dict[str, Any]]) -> str:
    """Build a re-investigation brief from detected conflicts.

    Args:
        conflicts: List of conflict dicts from conflict_scan.
        combs: All parsed combs for context.

    Returns:
        Reconcile prompt brief text.
    """
    lines = [
        "# RECONCILE re-investigation — conflict-resolution pass",
        "",
        "The conflicts below were detected. Investigate only what is needed to resolve the contradiction, narrowed.",
        "You MUST judge **execution reachability** (\"does that code actually run?\").",
        "",
    ]

    for i, conflict in enumerate(conflicts, 1):
        lines.append(f"## Conflict {i}: {conflict['type']}")
        lines.append(f"- Axis A: {conflict['axis_a']}")
        if conflict.get('axis_b'):
            lines.append(f"- Axis B: {conflict['axis_b']}")
        lines.append(f"- Detail: {conflict['detail']}")
        lines.append(f"- Evidence A: {conflict.get('evidence_a', 'N/A')}")
        if conflict.get('evidence_b'):
            lines.append(f"- Evidence B: {conflict['evidence_b']}")
        lines.append("")

        # Add specific narrowed questions based on conflict type
        if conflict["type"] == "root_cause_mismatch":
            lines.append(
                "**Narrowed question:** Of the two root causes above, which one is the "
                "path actually executed in the target scenario? Open both file:line "
                "locations and judge reachability via branch conditions, SQL WHERE, "
                "and try/except."
            )
        elif conflict["type"] == "termination_divergence":
            lines.append(
                "**Narrowed question:** One side left it resolved, the other unresolved. "
                "Is the code path flagged by the unresolved side actually reachable in "
                "the target scenario? Follow the file:line and judge the branch "
                "conditions statically."
            )
        elif conflict["type"] == "unresolved_conditional":
            lines.append(
                "**Narrowed question:** For the path left as reachable=conditional, does a "
                "scenario that actually reaches it exist? Pin down statically the concrete "
                "data state that satisfies the condition (e.g. a path where a row with a "
                "specific column being NULL exists)."
            )
        lines.append("")

    return '\n'.join(lines)


def run_reconcile_loop(
    combs: list[dict[str, Any]],
    comb_files: dict[str, str],
    seed_text: str,
    codebase_root: str,
    workdir: str,
    comb_contract: str,
    model: str = "gpt-5-mini",
    round_cap: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Run the reconcile loop until conflicts converge or round cap is reached.

    Args:
        combs: List of all parsed comb dicts.
        comb_files: Dict mapping axis_id -> comb file path.
        seed_text: Original seed text.
        codebase_root: Target codebase root.
        workdir: Working directory.
        comb_contract: Comb contract template.
        model: Model for copilot.
        round_cap: Maximum reconcile rounds (recipe default: 2).

    Returns:
        Tuple of (final_combs, remaining_conflicts, rounds_used).
    """
    combs_dir = os.path.join(workdir, "combs")
    os.makedirs(combs_dir, exist_ok=True)

    current_combs = list(combs)
    round_num = 0
    conflicts = scan_conflicts(current_combs)

    while conflicts and round_num < round_cap:
        round_num += 1
        logger.info("Reconcile round %d/%d — %d conflicts detected",
                     round_num, round_cap, len(conflicts))

        # Build reconcile brief
        brief = build_reconcile_brief(conflicts, current_combs)

        # Build reconcile prompt (using comb contract as base)
        reconcile_id = f"RECONCILE{round_num}" if round_num > 1 else "RECONCILE_R1"
        prompt = f"""{comb_contract}

[Assigned axis]
- axis_id: {reconcile_id}
- title: Reconcile re-investigation (round {round_num})
- brief: Resolve the conflicts below via a narrowed static investigation.

{brief}

[Original seed]
{seed_text}
"""

        # Run reconcile worker
        logger.info("  [reconcile] Launching worker for round %d...", round_num)
        comb_path = os.path.join(combs_dir, f"comb_{reconcile_id}.txt")
        err_path = os.path.join(combs_dir, f"err_{reconcile_id}.txt")

        try:
            result = subprocess.run(
                [shutil.which("copilot.cmd") or shutil.which("copilot") or "copilot", "--allow-all", "--model", model],
                input=prompt,  # prompt via stdin (cmd.exe argv truncates long/Korean prompts)
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=codebase_root,
                timeout=600,
            )

            with open(comb_path, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            with open(err_path, 'w', encoding='utf-8') as f:
                f.write(result.stderr)

            # Parse the reconcile comb
            reconcile_comb = extract_first_json(result.stdout)
            current_combs.append(reconcile_comb)
            comb_files[reconcile_id] = comb_path

            logger.info("  [reconcile] Round %d comb parsed: termination=%s",
                        round_num, reconcile_comb.get("termination", "?"))

        except (subprocess.TimeoutExpired, ValueError) as e:
            logger.error("  [reconcile] Round %d failed: %s", round_num, e)
            break

        # Re-scan for remaining conflicts
        conflicts = scan_conflicts(current_combs)
        if not conflicts:
            logger.info("  [reconcile] All conflicts resolved in round %d!", round_num)

    if conflicts:
        logger.warning("Reconcile: %d conflicts remain after %d rounds (hard boundary)",
                       len(conflicts), round_num)
        # Classify remaining as hard boundaries
        for c in conflicts:
            c["hard_boundary"] = True
            c["boundary_reason"] = f"Round cap {round_cap} reached without convergence"

    return current_combs, conflicts, round_num
