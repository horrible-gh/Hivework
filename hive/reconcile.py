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
  - Always ask about execution reachability ("이 코드가 진짜 도나?")
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
        "# RECONCILE 재조사 — 충돌 해소 조사",
        "",
        "아래 충돌이 발견됐다. 이 모순을 해소하는 데 필요한 것만 좁혀 조사하라.",
        "반드시 **실행 도달성**(\"그 코드가 진짜 도나?\")을 따져라.",
        "",
    ]

    for i, conflict in enumerate(conflicts, 1):
        lines.append(f"## 충돌 {i}: {conflict['type']}")
        lines.append(f"- 축 A: {conflict['axis_a']}")
        if conflict.get('axis_b'):
            lines.append(f"- 축 B: {conflict['axis_b']}")
        lines.append(f"- 상세: {conflict['detail']}")
        lines.append(f"- 근거 A: {conflict.get('evidence_a', 'N/A')}")
        if conflict.get('evidence_b'):
            lines.append(f"- 근거 B: {conflict['evidence_b']}")
        lines.append("")

        # Add specific narrowed questions based on conflict type
        if conflict["type"] == "root_cause_mismatch":
            lines.append(
                "**좁힌 질문:** 위 두 근본원인 중 어느 쪽이 대상 시나리오에서 "
                "실제로 실행되는 경로인가? 두 file:line을 모두 열어 분기조건·"
                "SQL WHERE·try/except를 따져 도달성을 판별하라."
            )
        elif conflict["type"] == "termination_divergence":
            lines.append(
                "**좁힌 질문:** 한쪽은 resolved, 다른 쪽은 미결로 남겼다. "
                "미결 측이 지적한 코드 경로가 실제로 대상 시나리오에서 "
                "실행 가능한가? file:line을 따라 분기조건을 정적으로 판별하라."
            )
        elif conflict["type"] == "unresolved_conditional":
            lines.append(
                "**좁힌 질문:** reachable=conditional로 남긴 경로에 "
                "실제로 도달하는 시나리오가 존재하는가? "
                "해당 조건을 만족시키는 구체적 데이터 상태(예: "
                "특정 칼럼이 NULL인 행이 존재하는 경로)를 정적으로 특정하라."
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

[배정된 축]
- axis_id: {reconcile_id}
- title: Reconcile 재조사 (라운드 {round_num})
- brief: 아래 충돌을 좁힌 정적 조사로 해소하라.

{brief}

[원본 시드]
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
