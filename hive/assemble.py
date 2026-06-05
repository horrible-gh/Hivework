"""Assemble stage — delegates final honey synthesis to a drone worker.

Takes all parsed combs + reconcile results + recipe §3 (Exit) rules
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
import logging
from typing import Any

from hive.providers import call_worker

logger = logging.getLogger("hive.assemble")

ASSEMBLE_SYSTEM = """# ROLE: ASSEMBLER (honey builder) — automated pipeline final stage

You are the ASSEMBLER in an automated Hivework pipeline. You receive:
1. All parsed comb results (drone investigation outputs) as JSON
2. Reconcile round results (if any)
3. The recipe §3 exit rules defining the output format

Your ONLY job is to synthesize all comb findings into a single honey markdown document.

## Output format (STRICT — honey_v2 skeleton):

1. **Header block**: investigation ID / project / date / investigator (Hivework drone pool) / status (investigation-only) / axis composition
2. **One-line key conclusion**: summarize the single most important finding in one sentence (> ⚠️ block)
3. **Numbered sections** = axes 1:1: for each axis, a §number, axis title, call-chain code-block trace, finding summary
4. **Gap analysis table**: | Gap | file:line | issue | status |
5. **Design SSOT comparison table**: | design ID §section | intent | actual code | match |
6. **Source table + conflict-convergence results**: | conclusion row | axis/drone | evidence file:line | + conflict-convergence narrative
7. **repro hypothesis + Fix direction**: minimal reproduction sequence + Fix option A/B/C + recommendation

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
    assemble_system: str | None = None,
) -> str:
    """Build the full prompt for the assemble worker.

    Args:
        combs: All parsed comb dicts (including reconcile combs).
        conflicts: Remaining conflicts (hard boundaries).
        recipe_section3: Recipe §3 output format rules.
        seed_text: Original seed text.
        rounds_used: Number of reconcile rounds used.
        assemble_system: Recipe-supplied assembler system prompt. When provided,
            it replaces the default investigation ``ASSEMBLE_SYSTEM`` so a recipe
            (e.g. a digest recipe) can drive a different output shape. When None,
            the default investigation assembler is used.

    Returns:
        Full prompt string for copilot.
    """
    system = assemble_system or ASSEMBLE_SYSTEM
    combs_json = json.dumps(combs, indent=2, ensure_ascii=False)
    conflicts_json = json.dumps(conflicts, indent=2, ensure_ascii=False) if conflicts else "[]"

    return f"""{system}

## Recipe §3 (Exit — output-format rules):
{recipe_section3}

## Conflict re-investigation results:
- Rounds used: {rounds_used}
- Remaining conflicts (hard boundaries): {conflicts_json}

## Original seed:
{seed_text}

## All comb results (JSON):
{combs_json}
"""


def _extract_recipe_section3(recipe_path: str) -> str:
    """Extract the §3 (Exit — honey output shape) section from the recipe card."""
    with open(recipe_path, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.split('\n')
    in_section = False
    section_lines = []
    for line in lines:
        if line.startswith('## ③') or '§3' in line:
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


def _extract_assemble_override(recipe_path: str) -> str | None:
    """Return a recipe-supplied assembler system prompt, if the recipe has one.

    A recipe may replace the default investigation assembler with its own by
    including a heading whose text is ``ASSEMBLE SYSTEM OVERRIDE`` (any heading
    level, case-insensitive) followed by a fenced code block holding the
    override system prompt. Investigation recipes omit the section and fall back
    to the default ``ASSEMBLE_SYSTEM``; a digest recipe supplies a digest
    assembler so the same pipeline produces a digest, not an investigation.

    Returns the override text, or None when absent/malformed.
    """
    with open(recipe_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    n = len(lines)
    i = 0
    while i < n:
        stripped = lines[i].lstrip()
        if (stripped.startswith('#')
                and stripped.lstrip('#').strip().lower() == 'assemble system override'):
            break
        i += 1
    else:
        return None

    # Advance to the opening fence (bail if another heading intervenes).
    i += 1
    while i < n and not lines[i].lstrip().startswith('```'):
        if lines[i].lstrip().startswith('#'):
            return None
        i += 1
    if i >= n:
        return None

    i += 1  # past the opening fence
    body: list[str] = []
    while i < n and not lines[i].lstrip().startswith('```'):
        body.append(lines[i])
        i += 1

    text = '\n'.join(body).strip()
    return text or None


def run_assemble(
    combs: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    seed_text: str,
    recipe_path: str,
    codebase_root: str,
    output_path: str,
    model: str = "gpt-5-mini",
    rounds_used: int = 0,
    provider: str = "copilot",
    ledger=None,
    provider_kwargs: dict | None = None,
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
    assemble_system = _extract_assemble_override(recipe_path)
    if assemble_system:
        logger.info("Using recipe-supplied assemble system override (%d chars)",
                    len(assemble_system))
    prompt = build_assemble_prompt(
        combs, conflicts, recipe_section3, seed_text, rounds_used,
        assemble_system=assemble_system,
    )

    logger.info("Running assemble worker...")
    logger.debug("Prompt length: %d chars", len(prompt))

    call_id = ledger.begin_call("assemble", "assemble", provider, model, prompt) \
        if ledger is not None else None
    try:
        wr = call_worker(provider, model, prompt, cwd=codebase_root, timeout=600,
                         **(provider_kwargs or {}))
    except Exception as e:  # timeout / provider error — record the failed row, then re-raise
        if ledger is not None:
            ledger.finish_call(call_id, output="", latency_s=0.0, ok=False,
                               err=str(e)[:200])
        raise
    raw_output = wr.stdout
    if ledger is not None:
        ledger.finish_call(call_id, output=wr.stdout, latency_s=wr.latency_s,
                           ok=wr.exit_code == 0,
                           err=wr.stderr[:200] if wr.exit_code != 0 else "",
                           real_tokens=wr.real_tokens)

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
    """Legacy wrapper around the provider adapter."""
    return call_worker("copilot", model, prompt, cwd=cwd, timeout=600).stdout
