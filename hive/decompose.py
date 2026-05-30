"""Decompose stage — delegates seed + recipe §1 to a drone worker.

Sends the seed instruction + recipe's fixed-axis rules to a copilot worker,
which returns the decomposition JSON: axes (A~G + fixed axes from recipe).

The decompose prompt template embeds:
  - The decomposer system prompt (from smoke/decompose_prompt.md)
  - The recipe §1 fixed-axis instructions
  - The seed (raw investigation instruction)

Output: list of axis dicts [{axis_id, title, brief, depends_on}]
"""

import json
import subprocess
import logging
import os
from typing import Any

from hive.parse import extract_first_json

logger = logging.getLogger("hive.decompose")

# Fixed axes that recipe_code_bug.md §1 requires always present
RECIPE_FIXED_AXES = """
## 반드시 포함할 고정 축 (레시피 §1):
- **설계 SSOT grep**: Documents/projects/<proj>/ 설계문서에서 "의도된 사양" grep. 코드 vs 설계 대조.
- **head-resolver / SQL 게이트**: 분기·루프·early-return·SQL WHERE 조건이 실행 도달성을 가르는 지점 조사. 특히 server/sql/queries/*.json.
- **리그레션 blame**: git log/git blame으로 증상 관련 라인의 최근 변경·도입 커밋 특정. "언제부터 깨졌나 / 이미 고쳐졌나".

이 세 축은 날것 A~G 외에 반드시 추가/보강하라.
"""

DECOMPOSE_SYSTEM = """# ROLE: DECOMPOSER ("queen") — automated fan-out pipeline

You are the DECOMPOSER in an automated pipeline. A human gives you ONE raw
instruction in natural language. Your ONLY job is to CUT it into independent
micro-tasks that free-tier workers ("drones") can each finish in a single
session, each producing a ~1 page brief.

You do NOT perform the research or work yourself.
You do NOT use any tools, read any files, or run any commands.
You ONLY output the decomposition.

## Rules for a good cut

1. Each micro-task = one worker, one session, ~1 page output. Self-contained:
   a worker who sees ONLY that task's brief (not the others) can do it.
2. Axes must be MUTUALLY EXCLUSIVE (no two workers research the same thing)
   and COLLECTIVELY EXHAUSTIVE (together they cover the whole ask).
3. If a task needs the OUTPUT of other tasks (e.g. synthesis / comparison /
   final integration), it is NOT parallel — place it in a LATER step and list
   its dependencies.
4. THRESHOLD GATE: if the natural number of parallel leaf tasks is < 5, do NOT
   fan out — return a single task and set fanout_decision = "single".
5. Each task carries: id, one-line title, a 2-4 line concrete brief
   (what to find / scope / deliverable), and depends_on (list of ids).

## Output format (STRICT) — return ONLY this JSON, no prose, no markdown fence:

{
  "fanout_decision": "fanout" | "single",
  "reason": "one line: why this many axes / why this split",
  "steps": [["A","B","C"], ["F"]],
  "tasks": [
    {"id": "A", "title": "...", "brief": "...", "depends_on": []}
  ]
}

`steps` = ordered list; each inner list is a set of task ids runnable in
parallel. Lists run in sequence (step 2 may use step 1's outputs).
"""


def build_decompose_prompt(seed_text: str, recipe_section1: str = "") -> str:
    """Build the full prompt for the decompose worker.

    Args:
        seed_text: The raw seed instruction text.
        recipe_section1: Recipe §1 fixed-axis rules (if available).

    Returns:
        Full prompt string for copilot.
    """
    recipe_part = recipe_section1 if recipe_section1 else RECIPE_FIXED_AXES
    return f"""{DECOMPOSE_SYSTEM}

{recipe_part}

## USER'S RAW INSTRUCTION:
{seed_text}
"""


def run_decompose(
    seed_text: str,
    recipe_path: str | None = None,
    codebase_root: str | None = None,
    model: str = "gpt-5-mini",
) -> dict[str, Any]:
    """Run the decompose stage by calling a copilot worker.

    Args:
        seed_text: The raw seed instruction.
        recipe_path: Path to recipe card (for extracting §1).
        codebase_root: Root path of target codebase.
        model: Model to use for copilot.

    Returns:
        Parsed decomposition JSON dict.
    """
    recipe_section1 = ""
    if recipe_path and os.path.exists(recipe_path):
        recipe_section1 = _extract_recipe_section1(recipe_path)

    prompt = build_decompose_prompt(seed_text, recipe_section1)
    logger.info("Running decompose worker...")
    logger.debug("Prompt length: %d chars", len(prompt))

    raw_output = _call_copilot(prompt, model=model, cwd=codebase_root)
    result = extract_first_json(raw_output)

    # Validate structure
    if "tasks" not in result:
        raise ValueError("Decompose output missing 'tasks' key")

    axes = result["tasks"]
    logger.info("Decompose produced %d axes: %s",
                len(axes), [a.get("id", "?") for a in axes])

    return result


def _extract_recipe_section1(recipe_path: str) -> str:
    """Extract §1 (입구 — 고정 절단축) section from recipe card."""
    with open(recipe_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract from "## ① 입구" to "## ②" or end
    lines = text.split('\n')
    in_section = False
    section_lines = []
    for line in lines:
        if '① 입구' in line or '§1' in line:
            in_section = True
            section_lines.append(line)
            continue
        if in_section:
            if line.startswith('## ②') or line.startswith('## ③'):
                break
            section_lines.append(line)

    if section_lines:
        return '\n'.join(section_lines)
    return RECIPE_FIXED_AXES  # Fallback


def _call_copilot(prompt: str, model: str = "gpt-5-mini",
                  cwd: str | None = None) -> str:
    """Call copilot CLI and return stdout.

    Args:
        prompt: Full prompt text.
        model: Model name.
        cwd: Working directory for subprocess.

    Returns:
        Raw stdout string.
    """
    cmd = [
        "copilot",
        "-p", prompt,
        "--allow-all",
        "--model", model,
    ]

    logger.debug("Calling copilot: model=%s, cwd=%s", model, cwd)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=300,  # 5 min timeout
    )

    if result.returncode != 0:
        logger.warning("Copilot returned code %d, stderr: %s",
                       result.returncode, result.stderr[:500])

    return result.stdout
