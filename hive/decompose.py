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
import re
import subprocess
from collections import Counter
from typing import Any

from hive.parse import extract_first_json
from hive.providers import call_worker
from hive.retriever import _ripgrep
from hive.searchplan import extract_keywords

logger = logging.getLogger("hive.decompose")

# Conditional FE-derived-state axis. This is deliberately opt-in: every leaf
# reaches the paid judge, so a generic "also inspect the frontend" axis would
# turn a targeted escalation into a permanent cost increase.
FE_DERIVED_AXIS_ID = "FE_DERIVED_STATE"
_FE_SOURCE_EXTS = frozenset({".ts", ".tsx", ".js", ".jsx", ".vue"})
_FE_ROOT_NAMES = frozenset({"client", "frontend", "web", "ui"})
_FE_DERIVED_KEYWORDS = (
    "buildStepStates", "headIndex", "workflowSteps", "stepStates",
    "nextStepIndex", "headType", "StepVisual", "indexOf",
    "currentStep", "activeStep",
)

_FE_POSITION_RE = re.compile(
    r"\boff[- ]by[- ]one\b"
    r"|\bone\s+(?:slot|step|stage|position)\s+(?:off|ahead|behind|shifted)\b"
    r"|\b(?:shifted|offset|misaligned)\s+(?:by\s+)?one\b"
    r"|\bwrong\s+(?:step|stage)\s+(?:is\s+)?(?:current|active|highlighted)\b"
    r"|\bindex(?:ing)?\s+(?:is\s+)?(?:wrong|off|shifted)\b"
    r"|\b(?:current|active)\s+(?:step|stage).{0,20}"
    r"(?:color|highlight).{0,12}(?:wrong|incorrect)\b"
    r"|\b(?:view[- ]?state|derived\s+(?:step\s+)?state).{0,20}"
    r"(?:wrong|incorrect|stale|mismatch)\b"
    r"|한\s*(?:칸|단계)\s*(?:씩\s*)?(?:밀림|밀려|어긋|차이)"
    r"|(?:현재|완료|다음)\s*(?:단계|스텝).{0,12}(?:밀림|어긋|잘못)"
    r"|뷰\s*상태.{0,12}(?:잘못|오류|불일치|낡)"
    r"|(?:앞|뒤)\s*(?:단계|스텝).{0,8}(?:표시|강조)",
    re.IGNORECASE,
)
_FE_PRESENTATION_RE = re.compile(
    r"\b(?:workflow|progress|step|stage)\s+(?:head|state|status|indicator|bar)\b"
    r"|\b(?:current|active)\s+(?:step|stage)\b"
    r"|\b(?:head|step|stage).{0,24}\b(?:color|highlight|current|active)\b"
    r"|\bview[- ]?state\b"
    r"|워크플로(?:우)?.{0,12}(?:헤드|head|단계|스텝|상태)"
    r"|(?:현재|완료|다음)\s*(?:단계|스텝)"
    r"|(?:단계|스텝).{0,12}(?:색|강조|상태)",
    re.IGNORECASE,
)
_FE_STATE_TRIO_RE = re.compile(
    r"\b(?:done|completed).{0,20}(?:current|active).{0,20}(?:future|pending)\b"
    r"|완료.{0,20}현재.{0,20}(?:미래|다음|예정)",
    re.IGNORECASE,
)


def is_fe_derived_state_symptom(seed_text: str) -> bool:
    """Whether the symptom warrants a paid FE derived-state investigation axis.

    The gate requires positional/state-derivation evidence plus a workflow/UI
    presentation cue. A generic HTTP/data-source report therefore stays on its
    existing axes even when it mentions a UI field such as ``workflow_head_type``.
    """
    text = seed_text or ""
    presentation = bool(_FE_PRESENTATION_RE.search(text))
    derived_error = bool(
        _FE_POSITION_RE.search(text) or _FE_STATE_TRIO_RE.search(text))
    return presentation and derived_error


def _frontend_source_globs(code_root: str | None, max_globs: int = 4) -> list[str]:
    """Return bounded FE source globs rooted only in frontend trees that exist.

    Prefer exact workflow/state producer and renderer files visible in the free
    repo tree. Fall back to an extension glob only when filenames provide no
    useful derived-state clue.
    """
    if not code_root:
        return []
    files = _git_tracked_files(code_root) or _walk_files(code_root)
    roots: dict[str, set[str]] = {}
    source_files: list[str] = []
    for raw in files:
        path = raw.replace("\\", "/").lstrip("./")
        if os.path.splitext(path)[1].lower() not in _FE_SOURCE_EXTS:
            continue
        parts = path.split("/")
        if not parts or parts[0].lower() not in _FE_ROOT_NAMES:
            continue
        try:
            src_idx = next(i for i, part in enumerate(parts)
                           if part.lower() == "src")
        except StopIteration:
            continue
        root = "/".join(parts[:src_idx + 1])
        roots.setdefault(root, set()).add(os.path.splitext(path)[1].lower())
        source_files.append(path)

    globs: list[str] = []

    def _score(path: str) -> int:
        low = path.lower()
        return (
            5 * ("workflow" in low)
            + 5 * ("viewstate" in low or "view_state" in low)
            + 3 * ("step" in low)
            + 2 * ("progress" in low)
            + 1 * ("state" in low)
            + 1 * ("head" in low)
        )

    logic = [p for p in source_files
             if os.path.splitext(p)[1].lower() != ".vue" and _score(p) > 0]
    renderers = [p for p in source_files
                 if os.path.splitext(p)[1].lower() == ".vue" and _score(p) > 0]
    logic.sort(key=lambda p: (-_score(p), p.lower()))
    renderers.sort(key=lambda p: (-_score(p), p.lower()))
    for path in logic[:2] + renderers[:2]:
        if path not in globs:
            globs.append(path)
        if len(globs) >= max_globs:
            return globs

    # No revealing filename: retain locality at the real FE src root and use
    # extensions that actually exist there. TS and Vue lead because producer and
    # renderer commonly split across those file classes.
    ext_order = (".ts", ".vue", ".tsx", ".js", ".jsx")
    for root in sorted(roots):
        for ext in ext_order:
            if ext == ".vue" and renderers:
                continue
            if ext != ".vue" and logic:
                continue
            if ext in roots[root]:
                globs.append(f"{root}/**/*{ext}")
                if len(globs) >= max_globs:
                    return globs
    return globs


def _task_covers_fe_derived_state(task: dict[str, Any]) -> bool:
    """True when a queen task already represents the conditional FE axis."""
    if task.get("depends_on"):
        return False
    if str(task.get("id", "")).upper() == FE_DERIVED_AXIS_ID:
        return True
    sp = task.get("search_plan") or {}
    if not isinstance(sp, dict):
        sp = {}
    blob = " ".join([
        str(task.get("id", "")), str(task.get("title", "")),
        str(task.get("brief", "")),
        *[str(x) for x in (sp.get("file_globs") or [])],
        *[str(x) for x in (sp.get("keywords") or [])],
    ]).lower()
    has_fe_scope = any(token in blob for token in (
        "client/", "frontend/", "web/src/", "ui/src/", ".vue", ".ts",
    ))
    has_derived_logic = any(token in blob for token in (
        "view state", "view-state", "derived state", "step state", "headindex",
        "indexof", "buildstepstates", "current step", "active step",
    ))
    return has_fe_scope and has_derived_logic


def ensure_fe_derived_state_axis(result: dict[str, Any], seed_text: str,
                                 code_root: str | None) -> dict[str, Any]:
    """Conditionally ensure one FE derived-state leaf exists in decomposition.

    Mutates and returns ``result``. Existing matching axes are enriched in place
    so the paid axis count does not grow. If the queen omitted the class entirely,
    one leaf is prepended so later position-based caps cannot bury it.
    """
    if not is_fe_derived_state_symptom(seed_text):
        return result

    tasks = result.get("tasks")
    if not isinstance(tasks, list):
        return result
    globs = _frontend_source_globs(code_root)

    for task in tasks:
        if not isinstance(task, dict) or not _task_covers_fe_derived_state(task):
            continue
        sp = task.get("search_plan")
        if not isinstance(sp, dict):
            sp = {}
            task["search_plan"] = sp
        existing_kw = [str(x) for x in (sp.get("keywords") or [])]
        sp["keywords"] = list(dict.fromkeys(
            list(_FE_DERIVED_KEYWORDS[:6]) + existing_kw
            + list(_FE_DERIVED_KEYWORDS[6:])))[:10]
        if globs:
            sp["file_globs"] = list(dict.fromkeys(
                [str(x) for x in (sp.get("file_globs") or [])] + globs))
        sp.setdefault("doc_topics", [])
        task.setdefault("coverage_risk", "thin")
        logger.info("FE derived-state symptom: reinforced existing axis %s",
                    task.get("id", "?"))
        return result

    axis = {
        "id": FE_DERIVED_AXIS_ID,
        "title": "Frontend derived workflow/view state",
        "brief": (
            "Trace the frontend logic that derives done, current, and future or active "
            "step state from the workflow head/index. Check index lookup, boundary "
            "and off-by-one handling, then follow the derived state into the "
            "rendering component; report the exact producer, not merely the view."
        ),
        "depends_on": [],
        "coverage_risk": "thin",
        "search_plan": {
            # With no real FE source scope, leave FIND empty so this axis stays
            # honestly thin instead of whole-tree matching backend names.
            "keywords": list(_FE_DERIVED_KEYWORDS) if globs else [],
            "file_globs": globs,
            "doc_topics": [],
        },
    }
    tasks.insert(0, axis)
    steps = result.get("steps")
    if isinstance(steps, list):
        if steps and isinstance(steps[0], list):
            steps[0].insert(0, FE_DERIVED_AXIS_ID)
        else:
            steps.insert(0, [FE_DERIVED_AXIS_ID])
    logger.info("FE derived-state symptom: injected conditional axis %s "
                "(frontend globs=%s)", FE_DERIVED_AXIS_ID, globs)
    return result

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

# ── Literal pre-grep given to the queen as BAIT, not a menu (M012) ─────────────
# The seed sometimes carries hard literals — error-stack identifiers, quoted
# strings, snake_case/CamelCase symbols, paths. Those are the one class of token
# we can resolve to real code for FREE, before any model runs (extract_keywords
# invents nothing; it only lifts tokens the seed already wrote). We grep them and
# show the queen where each already lands.
#
# CRUX (M012 §4 — anchoring): this is a HEAD-START, never a constraint. The real
# fix keyword is usually NOT in the request ("정렬이 안 돼요" → the fix lives at
# `ORDER BY ... IS NULL`, a token the seed never wrote), and the queen's value is
# exactly that symptom→code translation (decompose rule 6). So the prompt framing
# below says "treat this as one extra clue, then STILL invent your own keywords",
# never "pick from this list". On a pure-symptom seed extract_keywords returns
# near-nothing → the block is empty → an honest no-op (M012's predicted outcome).
def build_literal_preview(seed_text: str, code_root: str | None,
                          *, max_literals: int = 12,
                          max_hits_per_literal: int = 3,
                          max_lines: int = 60,
                          max_files_per_literal: int = 50) -> str:
    """Render a bounded map of where the seed's SPECIFIC literal tokens appear.

    Free, local, deterministic, never raises. Returns "" when there is no code
    root, no literal in the seed, or none of the literals ground — the caller then
    omits the block (no anchoring on an empty bait).

    The bait must point at SPECIFIC sites (an error string, an identifier), so
    noise is dropped on two axes: (1) tiny / path-fragment tokens (``server/``,
    3-char tokens) carry no locating signal; (2) a literal that grounds to MANY
    files (> ``max_files_per_literal`` — ``module`` hit 160, ``NULL`` 143) is too
    common to anchor anything. Survivors are RANKED by file-spread ascending so the
    most specific literals (a quoted UI label in 1 file, a unique identifier in a
    few) lead and are never the ones the line cap trims. Hits are capped per
    literal AND globally: ``_ripgrep``'s ``--max-count`` is PER FILE, so without a
    total cap a common token floods (450 lines of ``server/`` matches once buried
    the one useful ``ko.ts`` hit before the line cap could fire).
    """
    if not code_root:
        return ""
    scored: list[tuple[int, str, list]] = []
    for lit in extract_keywords(seed_text)[:max_literals]:
        if len(lit) < 4 or lit.endswith("/"):
            continue
        try:
            hits = _ripgrep(lit, [], code_root, max_hits=max_hits_per_literal)
        except (OSError, subprocess.SubprocessError):
            hits = []
        n_files = len({h["file"] for h in hits})
        if not hits or n_files > max_files_per_literal:
            continue
        scored.append((n_files, lit, hits))
    scored.sort(key=lambda t: t[0])                       # most specific first
    sections: list[str] = []
    total = 0
    for _n, lit, hits in scored:
        room = max_lines - total
        if room <= 0:
            break
        shown = hits[:min(max_hits_per_literal, room)]
        lines = [f"  {h['file'].removeprefix('./')}:{h['line']}  "
                 f"{h['text'][:120]}" for h in shown]
        sections.append(f"{lit}:\n" + "\n".join(lines))
        total += len(lines)
    return "\n".join(sections)


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
7. Each task ALSO carries `coverage_risk` — your OWN honest confidence that the
   blind grep above will actually find this axis's evidence. Set it to `"thin"`
   when you are NOT sure (the symptom is vaguely worded, your keywords are few or
   generic, or you are guessing where the code lives); otherwise `"ok"`. This is a
   self-doubt flag, not a verdict: a downstream check confirms it against what the
   local FIND really retrieved and only reinforces the axes you flagged AND that
   came back empty. Flag honestly — over-flagging wastes a check, under-flagging
   lets a thin axis slip through unreinforced.
8. CONDITIONAL FE-DERIVED-STATE AXIS: only when the symptom says a workflow/progress
   head or current step is shifted/off-by-one, the wrong step is highlighted/colored,
   or done/current/future view state is derived incorrectly, include one frontend
   axis covering the state producer (`*.ts`/`*.tsx`/`*.js`) and its renderer
   (`*.vue`/templates). Do NOT add this axis for ordinary HTTP, API, database, or
   data-source symptoms. Every axis reaches a paid judge, so this condition is a
   cost gate, not an optional suggestion.

## Output format (STRICT) — return ONLY this JSON, no prose, no markdown fence:

{
  "fanout_decision": "fanout" | "single",
  "reason": "one line: why this many axes / why this split",
  "steps": [["A","B","C"], ["F"]],
  "tasks": [
    {"id": "A", "title": "...", "brief": "...", "depends_on": [],
     "coverage_risk": "ok" | "thin",
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
                           repo_tree: str = "", literal_preview: str = "") -> str:
    """Build the full prompt for the decompose worker.

    Args:
        seed_text: The raw seed instruction text.
        recipe_section1: Recipe §1 fixed-axis rules (if available).
        repo_tree: A bounded file map of the target repo (build_repo_tree), so the
            queen anchors file_globs on real paths instead of guessing.
        literal_preview: A free pre-grep of the seed's literal tokens
            (build_literal_preview), attached as a HEAD-START — never a menu.

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
    literal_part = ""
    if literal_preview:
        literal_part = f"""
═══════════ LITERAL PRE-GREP (free head-start — NOT a menu) ═══════════
We grepped the EXACT tokens the request itself contains and list where each one
already appears below. Treat this as ONE extra clue — then STILL generate your
own `keywords` from your understanding of the symptom (rule 6). The real fix
keyword (e.g. `ORDER BY`, `IS NULL`) is usually NOT written in the request:
invent it, do NOT just pick from this list.
{literal_preview}
══════════════════════════════════════════════════════════════════════
"""
    return f"""Your task RIGHT NOW: read the software investigation request below and break it into a parallel research plan, then reply with ONLY a JSON object. This is a real, concrete task — act on it immediately. Do not reply conversationally, do not say "I'm ready", do not ask what to do. The request is already here:

═══════════════ INVESTIGATION REQUEST (decompose THIS) ═══════════════
{seed_text}
══════════════════════════════════════════════════════════════════════
{tree_part}{literal_part}
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
    literal_preview = build_literal_preview(seed_text, codebase_root)
    prompt = build_decompose_prompt(seed_text, recipe_section1, repo_tree,
                                    literal_preview)
    logger.info("Running decompose worker...")
    logger.info("Prompt length: %d chars (seed=%d, recipe§1=%d, tree=%d, "
                "literals=%d)", len(prompt), len(seed_text),
                len(recipe_section1), len(repo_tree), len(literal_preview))
    if repo_tree:
        logger.info("repo tree: fed %d chars of real paths to the queen "
                    "(anchors file_globs on existing files)", len(repo_tree))
    if literal_preview:
        logger.info("literal pre-grep: fed %d chars of free head-start hits "
                    "to the queen (bait, not a menu — rule 6 intact)",
                    len(literal_preview))

    # Persist the exact prompt sent, so delivery problems are inspectable.
    try:
        with open(os.path.join(os.getcwd(), "decompose_prompt_last.txt"),
                  "w", encoding="utf-8") as f:
            f.write(prompt)
    except OSError:
        pass

    call_id = ledger.begin_call("queen", "decompose", provider, model, prompt) \
        if ledger is not None else None
    try:
        wr = call_worker(provider, model, prompt, cwd=codebase_root, timeout=300,
                         on_start=(lambda: ledger.mark_running(call_id))
                         if (ledger is not None and call_id is not None) else None,
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

    ensure_fe_derived_state_axis(result, seed_text, codebase_root)
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
