"""Assemble stage — delegates final honey synthesis to a drone worker.

Takes all parsed combs + reconcile results + recipe §3 (출구) rules
and sends them to a copilot worker for final honey markdown assembly.

The output must match the honey_v2_N150.md skeleton:
  - Header block (ID/date/investigator/status/investigation-only)
  - Numbered sections = axes 1:1
  - Call-chain code-block traces (file:line → arrows)
  - Gap analysis table
  - Design SSOT comparison table
  - Source table + conflict convergence results
  - repro hypothesis + Fix direction options A/B/C
"""

import json
import os
import subprocess
import shutil
import logging
from typing import Any

from hive.parse import extract_first_json

logger = logging.getLogger("hive.assemble")

ASSEMBLE_SYSTEM = """# ROLE: ASSEMBLER (honey 조립기) — automated pipeline final stage

You are the ASSEMBLER in an automated Hivework pipeline. You receive:
1. All parsed comb results (drone investigation outputs) as JSON
2. Reconcile round results (if any)
3. The recipe §3 출구 rules defining the output format

Your ONLY job is to synthesize all comb findings into a single honey markdown document.

## Output format (STRICT — honey_v2 골격):

1. **헤더블록**: 조사 ID / 프로젝트 / 날짜 / 조사자(Hivework drone pool) / 상태(investigation-only) / 축 구성
2. **핵심 결론 한 줄**: 가장 중요한 발견을 한 문장으로 요약 (> ⚠️ 블록)
3. **번호 섹션** = 축 1:1: 각 축별로 §번호, 축 제목, 콜체인 코드블록 트레이스, 발견 요약
4. **Gap 분석 표**: | Gap | file:line | issue | 상태 |
5. **설계 SSOT 대조표**: | 설계ID §절 | 의도 | 코드 실제 | 일치 |
6. **출처표 + 충돌수렴 결과**: | 결론행 | 축/drone | 근거 file:line | + 충돌수렴 서술
7. **repro 가설 + Fix direction**: 최소 재현 시퀀스 + Fix option A/B/C + 추천

## Rules:
- Use ONLY information from the provided combs. Do NOT invent or assume.
- Every claim must cite the source axis and file:line from comb evidence.
- Call-chain traces must be in code blocks with file:line → arrows.
- Output pure markdown. No JSON wrapping.
"""


def build_assemble_prompt(
    combs: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    recipe_section3: str,
    seed_text: str,
    rounds_used: int = 0,
) -> str:
    """Build the full prompt for the assemble worker.

    Args:
        combs: All parsed comb dicts (including reconcile combs).
        conflicts: Remaining conflicts (hard boundaries).
        recipe_section3: Recipe §3 output format rules.
        seed_text: Original seed text.
        rounds_used: Number of reconcile rounds used.

    Returns:
        Full prompt string for copilot.
    """
    combs_json = json.dumps(combs, indent=2, ensure_ascii=False)
    conflicts_json = json.dumps(conflicts, indent=2, ensure_ascii=False) if conflicts else "[]"

    return f"""{ASSEMBLE_SYSTEM}

## 레시피 §3 (출구 — 출력형태 규칙):
{recipe_section3}

## 충돌 재조사 결과:
- 라운드 사용: {rounds_used}
- 잔여 충돌(경성경계): {conflicts_json}

## 원본 시드:
{seed_text}

## 전체 comb 결과 (JSON):
{combs_json}
"""


def _extract_recipe_section3(recipe_path: str) -> str:
    """Extract §3 (출구 — honey 출력형태) section from recipe card."""
    with open(recipe_path, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.split('\n')
    in_section = False
    section_lines = []
    for line in lines:
        if '③ 출구' in line or '§3' in line:
            in_section = True
            section_lines.append(line)
            continue
        if in_section:
            # Stop at next top-level section or end
            if line.startswith('## ④') or line.startswith('## ⑤'):
                break
            section_lines.append(line)

    if section_lines:
        return '\n'.join(section_lines)

    # Fallback: return the entire recipe as context
    return text


def run_assemble(
    combs: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    seed_text: str,
    recipe_path: str,
    codebase_root: str,
    output_path: str,
    model: str = "gpt-5-mini",
    rounds_used: int = 0,
) -> str:
    """Run the assemble stage by calling a copilot worker.

    Args:
        combs: All parsed comb dicts.
        conflicts: Remaining conflicts (hard boundaries).
        seed_text: Original seed text.
        recipe_path: Path to recipe card.
        codebase_root: Target codebase root.
        output_path: Path to write the honey markdown.
        model: Model for copilot.
        rounds_used: Number of reconcile rounds used.

    Returns:
        Path to the written honey file.
    """
    recipe_section3 = _extract_recipe_section3(recipe_path)
    prompt = build_assemble_prompt(
        combs, conflicts, recipe_section3, seed_text, rounds_used
    )

    logger.info("Running assemble worker...")
    logger.debug("Prompt length: %d chars", len(prompt))

    raw_output = _call_copilot(prompt, model=model, cwd=codebase_root)

    # The assemble worker outputs markdown directly (not JSON)
    # Strip any leading tool-trace lines (● lines)
    honey = _strip_tool_traces(raw_output)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(honey)

    logger.info("Honey written to %s (%d bytes)", output_path, len(honey))
    return output_path


def _strip_tool_traces(raw: str) -> str:
    """Strip leading ● tool-trace lines from copilot output.

    The assemble worker shouldn't use tools, but just in case.
    """
    lines = raw.split('\n')
    # Find the first line that isn't a tool trace
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('●') or stripped.startswith('✗'):
            start = i + 1
            continue
        if stripped.startswith('│') or stripped.startswith('└'):
            start = i + 1
            continue
        if not stripped:
            # Blank line between traces — keep scanning
            if i > 0 and start == i:
                start = i + 1
                continue
        break

    return '\n'.join(lines[start:])


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
    # Mirror ai_launcher run_worker.py: copilot reads the prompt from STDIN
    # (no -p arg). Passing a long/non-ASCII prompt as a cmd.exe argv truncates
    # it (cmd.exe ~8KB limit + cp932 codepage mangling of Korean). Stdin avoids both.
    cmd = [
        shutil.which("copilot.cmd") or shutil.which("copilot") or "copilot",
        "--allow-all",
        "--model", model,
    ]

    logger.debug("Calling copilot: model=%s, cwd=%s", model, cwd)
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        timeout=600,  # 10 min timeout for assemble
    )

    if result.returncode != 0:
        logger.warning("Copilot returned code %d, stderr: %s",
                       result.returncode, result.stderr[:500])

    return result.stdout
