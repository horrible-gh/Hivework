"""Decompose stage — delegates seed + recipe §1 to a drone worker.

Sends the seed instruction + recipe's fixed-axis rules to a copilot worker,
which returns the decomposition JSON: axes (A~G + fixed axes from recipe).

The decompose prompt template embeds:
  - The decomposer system prompt (from smoke/decompose_prompt.md)
  - The recipe §1 fixed-axis instructions
  - The seed (raw investigation instruction)

Output: list of axis dicts [{id, title, brief, depends_on, search_plan}]
where search_plan = {keywords, file_globs, doc_topics} is the blind seed the
local retriever (hive.searchplan bridge) lowers into a SearchPlan.
"""

import json
import logging
import os
import subprocess
from collections import Counter
from typing import Any

from hive.parse import extract_first_json
from hive.providers import call_worker

logger = logging.getLogger("hive.decompose")

# ── Repo file tree given to the queen so axes anchor on REAL paths ─────────────
# The decomposer fans out blind to the repo, so it guesses file_globs (wrong
# extensions, frontend-only or backend-only coverage) and the local FIND windows
# nothing (NR168: a frontend off-by-one whose Vue component the FE axis never
# located because the seed only named backend anchors). Feeding the actual file
# list — free, local, deterministic — lets the queen anchor globs on paths that
# exist and cover BOTH trees. No model, no cost.
_TREE_SKIP_EXT = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp4", ".mov", ".mp3", ".wav", ".pdf", ".zip", ".gz", ".tar", ".7z",
    ".lock", ".map",
})
_TREE_SKIP_DIR = (
    "node_modules/", ".git/", "dist/", "build/", "__pycache__/", ".venv/",
    "venv/", "vendor/", ".next/", "coverage/", ".idea/", ".vscode/",
)


def _git_tracked_files(code_root: str) -> list[str]:
    """Tracked files via ``git ls-files`` (respects .gitignore). [] on failure."""
    try:
        p = subprocess.run(
            ["git", "-C", code_root, "ls-files"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30)
        if p.returncode == 0:
            return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    except (subprocess.SubprocessError, OSError):
        pass
    return []


def _walk_files(code_root: str, cap: int = 20000) -> list[str]:
    """Fallback when not a git repo: os.walk, pruning common noise dirs."""
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
            "build", ".next", "vendor", "coverage", ".idea", ".vscode"}
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(code_root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        rel = os.path.relpath(dirpath, code_root).replace("\\", "/")
        for fn in filenames:
            out.append(fn if rel == "." else f"{rel}/{fn}")
            if len(out) >= cap:
                return out
    return out


def _tree_useful(path: str) -> bool:
    """Drop binary/noise paths so the tree stays a navigable source map."""
    p = path.lower()
    if any(p.startswith(d) or f"/{d}" in p for d in _TREE_SKIP_DIR):
        return False
    return os.path.splitext(p)[1] not in _TREE_SKIP_EXT


def build_repo_tree(code_root: str | None, max_lines: int = 400) -> str:
    """Render a bounded, deterministic file map of ``code_root`` for the queen.

    Lists tracked source files (``git ls-files``, else an os.walk fallback),
    filtered of binaries. Under ``max_lines`` → the file list verbatim; over it →
    a directory summary with file counts (still navigable, bounded). Free & local.
    """
    if not code_root:
        return ""
    files = _git_tracked_files(code_root) or _walk_files(code_root)
    files = sorted(f.replace("\\", "/") for f in files if _tree_useful(f))
    if not files:
        return ""
    if len(files) <= max_lines:
        return "\n".join(files)
    counts = Counter(os.path.dirname(f) or "." for f in files)
    dir_lines = [f"{d}/  ({n} files)" for d, n in sorted(counts.items())]
    if len(dir_lines) <= max_lines:
        return ("# large repo — directories with file counts (grep within these "
                "to confirm exact files):\n" + "\n".join(dir_lines))
    top = Counter(f.split("/", 1)[0] for f in files)
    return ("# very large repo — top-level areas with file counts:\n" +
            "\n".join(f"{d}/  ({n} files)" for d, n in sorted(top.items())))

# Fixed axes that recipe_code_bug.md §1 requires always present
RECIPE_FIXED_AXES = """
## Fixed axes that MUST be included (recipe §1):
- **Design SSOT grep**: grep the design docs under Documents/projects/<proj>/ for the "intended spec." Contrast code vs design.
- **head-resolver / SQL gate**: investigate where branches, loops, early-returns, and SQL WHERE conditions decide execution reachability. Especially server/sql/queries/*.json.
- **Regression blame**: use git log / git blame to pin the recent change / introducing commit for the lines tied to the symptom. "Since when did it break / is it already fixed."

In addition to the raw axes A-G, you MUST add/reinforce these three axes.
"""

DECOMPOSE_SYSTEM = """# ROLE: DECOMPOSER ("queen") — automated fan-out pipeline

You are the DECOMPOSER in an automated pipeline. A human gives you ONE raw
instruction in natural language. Your ONLY job is to CUT it into independent
micro-tasks that free-tier workers ("drones") can each finish in a single
session, each producing a ~1 page brief.

You do NOT perform the full investigation yourself — your deliverable is the
PLAN, not the findings. A REPO FILE TREE is provided below — anchor your
`search_plan.file_globs` on REAL paths from it (correct directories AND
extensions); do not guess paths. If the request can span frontend and backend,
make sure your axes cover BOTH trees. Then output the decomposition JSON.

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
6. Each task ALSO carries a `search_plan` — the BLIND seed a local grep will use
   to find that axis's evidence (a downstream FIND consumes it, no human reads
   it). Fill it from the AXIS ITSELF; never from any answer you happen to know:
   - `keywords`: 4-10 EXACT tokens a `grep` would hit — identifiers, symbol
     names, SQL tokens (e.g. `type_code`, `group_head`, `ORDER BY`, `IS NULL`),
     literal strings. NOT prose ("the head logic"); NOT generic words ("data").
   - `file_globs`: 1-4 path scopes (e.g. `server/sql/queries/*.json`,
     `server/modules/**/db/**/*.py`). Narrow to where the evidence lives.
   - `doc_topics`: design-doc topics/codes for design axes (e.g. `D030`,
     `head semantics`); [] for pure-code axes.

## Output format (STRICT) — return ONLY this JSON, no prose, no markdown fence:

{
  "fanout_decision": "fanout" | "single",
  "reason": "one line: why this many axes / why this split",
  "steps": [["A","B","C"], ["F"]],
  "tasks": [
    {"id": "A", "title": "...", "brief": "...", "depends_on": [],
     "search_plan": {"keywords": ["..."], "file_globs": ["..."], "doc_topics": ["..."]}}
  ]
}

`steps` = ordered list; each inner list is a set of task ids runnable in
parallel. Lists run in sequence (step 2 may use step 1's outputs).

[JSON validity] Emit STANDARD JSON. String values use plain double-quote
delimiters — do NOT backslash-escape the delimiters themselves. A keyword like
mode='next' is written `"mode='next'"`, NEVER `\\"mode='next'\\"`. Inside a
value, prefer single quotes so no escaping is needed.
"""


def build_decompose_prompt(seed_text: str, recipe_section1: str = "",
                           repo_tree: str = "") -> str:
    """Build the full prompt for the decompose worker.

    Args:
        seed_text: The raw seed instruction text.
        recipe_section1: Recipe §1 fixed-axis rules (if available).
        repo_tree: A bounded file map of the target repo (build_repo_tree), so the
            queen anchors file_globs on real paths instead of guessing.

    Returns:
        Full prompt string for copilot.
    """
    recipe_part = recipe_section1 if recipe_section1 else RECIPE_FIXED_AXES
    tree_part = ""
    if repo_tree:
        tree_part = f"""
═══════════════ REPO FILE TREE (these paths REALLY exist) ═══════════════
Anchor every `search_plan.file_globs` on a REAL path below — correct directory
AND extension. Do NOT guess paths. If the request spans frontend and backend,
ensure your axes cover BOTH.
{repo_tree}
══════════════════════════════════════════════════════════════════════
"""
    return f"""Your task RIGHT NOW: read the software investigation request below and break it into a parallel research plan, then reply with ONLY a JSON object. This is a real, concrete task — act on it immediately. Do not reply conversationally, do not say "I'm ready", do not ask what to do. The request is already here:

═══════════════ INVESTIGATION REQUEST (decompose THIS) ═══════════════
{seed_text}
══════════════════════════════════════════════════════════════════════
{tree_part}
Now cut that request into independent micro-tasks (axes) for free-tier worker
agents, following the rules and output schema below.

{DECOMPOSE_SYSTEM}

{recipe_part}

═══════════════════════════════════════════════════════════════════
Respond NOW with ONLY the JSON object specified above — start with `{{`, end
with `}}`. No prose, no markdown fences, no "I'm ready", nothing else.
"""


def run_decompose(
    seed_text: str,
    recipe_path: str | None = None,
    codebase_root: str | None = None,
    model: str = "gpt-5-mini",
    provider: str = "copilot",
    ledger=None,
    provider_kwargs: dict | None = None,
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

    repo_tree = build_repo_tree(codebase_root)
    prompt = build_decompose_prompt(seed_text, recipe_section1, repo_tree)
    logger.info("Running decompose worker...")
    logger.info("Prompt length: %d chars (seed=%d, recipe§1=%d, tree=%d)",
                len(prompt), len(seed_text), len(recipe_section1), len(repo_tree))
    if repo_tree:
        logger.info("repo tree: fed %d chars of real paths to the queen "
                    "(anchors file_globs on existing files)", len(repo_tree))

    # Persist the exact prompt sent, so delivery problems are inspectable.
    try:
        with open(os.path.join(os.getcwd(), "decompose_prompt_last.txt"),
                  "w", encoding="utf-8") as f:
            f.write(prompt)
    except OSError:
        pass

    wr = call_worker(provider, model, prompt, cwd=codebase_root, timeout=300,
                     **(provider_kwargs or {}))
    raw_output = wr.stdout
    if ledger is not None:
        ledger.record_call("queen", "decompose", provider, model,
                           prompt=prompt, output=wr.stdout, latency_s=wr.latency_s,
                           ok=wr.exit_code == 0,
                           err=wr.stderr[:200] if wr.exit_code != 0 else "",
                           real_tokens=wr.real_tokens)

    # Persist raw output so decompose failures are never blind (was: no dump).
    dump_path = os.path.join(os.getcwd(), "decompose_raw_last.txt")
    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            f.write(raw_output or "")
    except OSError:
        dump_path = "(dump failed)"

    if not raw_output or not raw_output.strip():
        raise ValueError(
            f"Decompose worker returned empty output (quota/timeout/rc!=0?). "
            f"Raw saved to {dump_path}"
        )

    try:
        result = extract_first_json(raw_output)
    except ValueError as e:
        snippet = raw_output.strip()[:800]
        raise ValueError(
            f"{e}\nWorker did not emit the required JSON object. "
            f"Full raw saved to {dump_path}. First 800 chars of output:\n{snippet}"
        ) from e

    # Validate structure
    if "tasks" not in result:
        raise ValueError("Decompose output missing 'tasks' key")

    axes = result["tasks"]
    logger.info("Decompose produced %d axes: %s",
                len(axes), [a.get("id", "?") for a in axes])

    return result


def _extract_recipe_section1(recipe_path: str) -> str:
    """Extract the §1 (Entrance — fixed cut axes) section from the recipe card."""
    with open(recipe_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract from "## ① Entrance" to "## ②"/"## ③" or end
    lines = text.split('\n')
    in_section = False
    section_lines = []
    for line in lines:
        if line.startswith('## ①') or '§1' in line:
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
    """Legacy wrapper around the provider adapter."""
    return call_worker("copilot", model, prompt, cwd=cwd, timeout=300).stdout
