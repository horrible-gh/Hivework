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
DEFAULT_COMB_CONTRACT = """[역할] 너는 Hivework 무료워커(drone) 1명. 배정된 단일 조사축 하나만 판다. 코드 수정 금지 — investigation-only. 모든 주장은 실제 파일을 grep/read로 열어 확인한 file:line 근거 필수. 추측 금지.

[대상 코드베이스 루트] {codebase_root} (git repo)

[깊이 규약 — 얕은 comb 금지] 다음을 반드시 한다:
1. **실행 도달성**: 코드가 "있다"가 아니라 "실제로 실행되는가"를 따져라. 분기조건·early return·try/except 삼킴·**SQL WHERE 게이트**가 그 블록을 스킵시키는지 확인. "존재"와 "도달"을 구분해 적어라.
2. **콜체인 트레이스**: 진입점→…→DB write까지 file:line을 `→`로 이어라.
3. **설계 대조**(가능하면): 코드 동작 vs 설계문서가 의도한 사양을 대조. 어긋나면 그게 버그, 일치하면 의도.
4. **blame**(축이 리그레션/이력이면): `git log`/`git blame`으로 관련 라인의 도입·수정 커밋(해시+제목)을 특정. "이미 고쳐졌나"도 확인.

[출력 규약 — comb] 아래 JSON 한 덩어리만. 산문·JSON 외 텍스트 금지.
{{
  "axis_id": "<축 id>",
  "axis_title": "<축 제목>",
  "trace": "<진입점→…→DB/UI까지 콜체인 한 줄, file:line 포함. 해당없으면 null>",
  "findings": [
    {{
      "claim": "<확인한 사실>",
      "evidence": [{{"file":"<상대경로>","lines":"<예 49-78>","what":"<그 라인이 보여주는 것>"}}],
      "reachable": "yes|no|conditional — 이 코드가 대상 시나리오에서 실제 실행되는가 + 조건",
      "confidence": "high|med|low"
    }}
  ],
  "design_ref": [{{"doc":"<설계ID 예 M026 §8-1>","intended":"<설계가 의도한 것>","matches_code":"yes|no"}}],
  "regression": {{"commit":"<해시 제목 / null>","what_changed":"<무엇이 언제 바뀜 / null>"}},
  "root_cause_signal": "<이 축이 증상의 근본원인을 직접 짚으면 file:line, 아니면 null>",
  "cross_refs": ["<다른 축 id>"],
  "termination": "resolved | needs_runtime | needs_external | needs_pm",
  "notes": "<한줄. 못 닫았으면 무엇을 더 봐야 하는지>"
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

[배정된 축]
- axis_id: {axis_id}
- title: {title}
- brief: {brief}

[원본 시드 (전체 맥락)]
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
