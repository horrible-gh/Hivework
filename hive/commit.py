"""Commit pipeline — author a commit plan, then execute it deterministically.

This mirrors the fix-extension's ``specify`` → ``apply`` split, applied to git:

  commit-plan (this, propose)   git working tree + contract → commit-plan JSON (SSOT)
        │                       ← single author, calls a model
        ▼
  commit (this, execute)        commit-plan JSON → git add/commit per commit
                                ← NO worker call, pure deterministic local stage

Two user-facing commands:

  * ``run_propose``  — a single author worker reads the live ``git status`` and the
    commit-plan contract, then groups the changes into atomic conventional commits.
    The output is a commit-plan JSON (the SSOT). It is propose-only: nothing is
    committed here.
  * ``run_commit``   — re-verifies the plan against the LIVE git state and renders a
    proposal table. Default is dry-run. With ``write=True`` and a READY plan it runs
    ``git add``/``git commit`` per planned commit, scoped to that commit's pathspecs,
    all-or-nothing (a mid-sequence failure soft-resets HEAD back to where it started).

Design notes (mirrored from hive/specify.py + hive/apply.py):
  - The contract file doubles as the author's role/system prompt (single SSOT for the
    authoring rules; no duplicated prompt text here).
  - Hivework is an independent tool: the commit policy is owned by Hivework's own
    contract, not loaded from any external rule document at runtime.
  - ``gate.commit`` in the plan is model-authored and is NEVER trusted to drive a
    commit — the write switch is the human-held ``--write`` flag.
  - The plan JSON is the SSOT; the proposal table is a DERIVED view rendered here.
  - Everything is re-verified against live git at commit time; drift is refused.
"""

import json
import logging
import os
import re
import subprocess
from typing import Any

from hive.parse import extract_first_json
from hive.providers import call_worker

logger = logging.getLogger("hive.commit")

# The authoring contract doubles as the commit author's role/system prompt.
_DEFAULT_CONTRACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "recipes", "commit_plan_contract_v1.md"
)

# Structural expectations for the emitted commit-plan JSON.
_REQUIRED_KEYS = ("commits", "gate", "termination")
_VALID_TERMINATION = {"ready_to_commit", "needs_pm"}

# Conventional-commit types Hivework's contract allows.
_ALLOWED_TYPES = ("feat", "fix", "docs", "chore", "refactor", "test", "style")
_MESSAGE_RE = re.compile(
    r"^(?:" + "|".join(_ALLOWED_TYPES) + r")(?:\([a-z0-9._\-/]+\))?: .+"
)

# When the plan leaves a staged path unassigned, the execute stage sweeps it into
# this final commit rather than halting — staging is explicit commit intent (§0).
_SWEEP_MESSAGE = "chore: commit staged changes not grouped by the plan"

# Per-commit applicability statuses (only "committable" can contribute to ready).
COMMITTABLE = "committable"
EMPTY_FILES = "empty_files"
BAD_MESSAGE = "bad_message"
FILE_NOT_CHANGED = "file_not_changed"
FILE_DUPLICATED = "file_duplicated"


# ── git helpers ──────────────────────────────────────────────────────────────

def _git(repo_root: str, args: list[str], check: bool = False) -> subprocess.CompletedProcess:
    """Run a git command in ``repo_root`` and return the CompletedProcess.

    stdout/stderr are decoded as UTF-8 (errors replaced) so non-ASCII paths and
    messages don't crash on Windows. With ``check=True`` a non-zero exit raises.
    """
    cmd = ["git", "-C", repo_root, *args]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (rc={result.returncode}): "
            f"{result.stderr.strip()}")
    return result


def changed_paths(repo_root: str) -> set[str]:
    """Return the set of paths with uncommitted changes (forward-slashed, rel).

    Covers staged, unstaged, and untracked files via ``git status --porcelain``.
    Renames (``R  old -> new``) contribute the new path. Quoted paths (paths with
    spaces/unicode under core.quotepath) are unquoted.
    """
    res = _git(repo_root, ["status", "--porcelain"], check=True)
    paths: set[str] = set()
    for raw in res.stdout.splitlines():
        if not raw.strip():
            continue
        # Porcelain v1: XY<space>path  (status code is 2 chars + a space)
        entry = raw[3:]
        # Renames/copies: "old -> new" — keep the destination path.
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        paths.add(_unquote_path(entry))
    return paths


def staged_paths(repo_root: str) -> set[str]:
    """Return paths currently staged in the index vs HEAD (forward-slashed, rel).

    Staged content is the PM's *explicit* intent to commit — it must not be silently
    dropped. Renames/copies contribute the destination path. Uses ``--name-status``
    against HEAD; on an unborn branch the index is compared against the empty tree, so
    the first-ever staged files are still reported.
    """
    res = _git(repo_root, ["diff", "--cached", "--name-status"], check=True)
    paths: set[str] = set()
    for raw in res.stdout.splitlines():
        if not raw.strip():
            continue
        # "X<TAB>path" or, for renames/copies, "Rxxx<TAB>old<TAB>new".
        rel = raw.split("\t")[-1]  # destination for R/C; the path otherwise
        paths.add(_unquote_path(rel))
    return paths


def _unquote_path(path: str) -> str:
    """Normalize a porcelain path: strip C-style quoting, use forward slashes."""
    path = path.strip()
    if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
        # git quotes paths with special chars; decode the C-escaped, UTF-8 bytes.
        try:
            path = path[1:-1].encode("latin-1", "backslashreplace").decode(
                "unicode_escape").encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            path = path[1:-1]
    return path.replace("\\", "/")


def _head_commit(repo_root: str) -> str | None:
    """Return the current HEAD sha, or None on an unborn branch (no commits yet)."""
    res = _git(repo_root, ["rev-parse", "HEAD"])
    return res.stdout.strip() if res.returncode == 0 else None


# ── propose (author the plan) ────────────────────────────────────────────────

def load_contract(contract_path: str | None = None) -> str:
    """Load the commit-plan authoring contract (the author's role prompt)."""
    path = contract_path or _DEFAULT_CONTRACT_PATH
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _git_context(repo_root: str) -> str:
    """Render the live git state given to the author (status + a stat summary)."""
    status = _git(repo_root, ["status", "--porcelain"], check=True).stdout
    staged = _git(repo_root, ["diff", "--cached", "--name-status"]).stdout
    stat = _git(repo_root, ["diff", "--stat", "HEAD"]).stdout
    untracked = _git(repo_root, ["status", "--short", "--untracked-files=all"]).stdout
    parts = ["[git status --porcelain]", status or "(clean)", ""]
    parts += ["[staged paths — git diff --cached --name-status]",
              "These are ALREADY staged = the PM's explicit intent to commit. "
              "Each MUST be assigned to a commit; never route a staged path to "
              "leftover. A staged deletion (D) of a generated file is a deliberate "
              "untracking — commit it as chore.",
              staged or "(nothing staged)", ""]
    if stat.strip():
        parts += ["[git diff --stat HEAD]", stat, ""]
    parts += ["[git status --short -uall]", untracked or "(none)"]
    return "\n".join(parts)


# Injected into the author prompt when the change set is too large to read each file
# economically — group by path/name alone, never open files (caps credit spend).
_FILENAME_ONLY_DIRECTIVE = """
[BUDGET MODE — filename-only grouping (large change set: {n_changed} files > {threshold})]
This change set is too large to read each file economically. Group the changes using
ONLY the file PATHS and NAMES shown in the git state above (directory, extension, and
naming conventions). Do NOT open, read, or diff any file — no file-content tools.
Infer each commit's purpose and conventional-commit type from the paths alone (e.g.
``docs/**`` or ``*.md`` → ``docs``; ``tests/**`` or ``test_*`` → ``test``; config/build
files → ``chore``). Consolidate aggressively by directory/topic rather than over-split.
Every changed path must still be assigned to exactly one commit. Emit the same
commit-plan JSON schema; ``ready_to_commit`` as usual.
"""


def build_propose_prompt(
    contract_text: str,
    repo_root: str,
    git_context: str,
    prev_plan_json: str | None = None,
    feedback: str | None = None,
    filename_only: bool = False,
    n_changed: int = 0,
    threshold: int = 0,
) -> str:
    """Build the full prompt for the single commit author.

    The contract is the role/system prompt; the live git state is the input to
    group. If the PM rejected a previous plan, the previous plan plus the PM's
    feedback are appended so the author revises rather than starting blind. When
    ``filename_only`` is set the change set is too large to read economically, so a
    budget directive is injected telling the author to group from paths alone.
    """
    prompt = f"""{contract_text}

[Repo root — the working tree to group into commits]
{repo_root}

[Live git state — group exactly these changed files, nothing else]
{git_context}
"""
    if filename_only:
        prompt += _FILENAME_ONLY_DIRECTIVE.format(
            n_changed=n_changed, threshold=threshold)
    if prev_plan_json:
        prompt += f"""
[Previous plan you proposed — the PM rejected it, revise it]
{prev_plan_json}
"""
    if feedback:
        prompt += f"""
[PM feedback — apply this when revising the plan]
{feedback}
"""
    prompt += "\nEmit the revised commit-plan JSON only.\n" if (prev_plan_json or feedback) \
        else "\nEmit the commit-plan JSON only.\n"
    return prompt


def _normalize_plan(plan: dict[str, Any], repo_root: str) -> dict[str, Any]:
    """Enforce invariants: gate.commit is always false; record repo_root."""
    gate = plan.get("gate")
    if not isinstance(gate, dict):
        gate = {}
        plan["gate"] = gate
    if gate.get("commit") is not False:
        logger.warning("commit-plan: gate.commit was %r — forcing false "
                       "(write is human-held)", gate.get("commit"))
        gate["commit"] = False
    plan.setdefault("repo_root", os.path.abspath(repo_root))
    return plan


def _validate_plan(plan: dict[str, Any]) -> list[str]:
    """Return a list of structural problems (empty = ok). Non-fatal; caller decides."""
    problems: list[str] = []
    for key in _REQUIRED_KEYS:
        if key not in plan:
            problems.append(f"missing required key: {key}")
    term = plan.get("termination")
    if term is not None and term not in _VALID_TERMINATION:
        problems.append(f"invalid termination: {term!r}")
    if "commits" in plan and not isinstance(plan["commits"], list):
        problems.append("commits is not a list")
    return problems


def run_propose(
    repo_root: str,
    output_path: str,
    contract_path: str | None = None,
    model: str = "claude-haiku-4.5",
    provider: str = "copilot",
    ledger=None,
    provider_kwargs: dict | None = None,
    prev_plan_path: str | None = None,
    feedback: str | None = None,
    filename_only_threshold: int = 0,
) -> dict[str, Any]:
    """Run the propose stage: live git working tree → commit-plan JSON (SSOT).

    Calls a single author worker, extracts the first complete JSON object from its
    stdout, forces ``gate.commit`` false, writes the plan to ``output_path``, and
    returns the parsed dict. Propose-only — nothing is committed.

    When ``filename_only_threshold`` > 0 and the number of changed paths exceeds it,
    a budget directive is injected so the author groups from file paths alone without
    opening files (caps credit spend on huge, commonly documentation, change sets).

    Raises:
        ValueError: if the author produced no parseable JSON object.
        RuntimeError: if ``repo_root`` is not a git work tree.
    """
    if not os.path.isdir(os.path.join(repo_root, ".git")) and \
            _git(repo_root, ["rev-parse", "--is-inside-work-tree"]).returncode != 0:
        raise RuntimeError(f"not a git work tree: {repo_root}")

    contract_text = load_contract(contract_path)
    git_context = _git_context(repo_root)

    n_changed = len(changed_paths(repo_root))
    filename_only = filename_only_threshold > 0 and n_changed > filename_only_threshold
    if filename_only:
        logger.info("Large change set (%d files > threshold %d): filename-only "
                    "grouping — the author will NOT open files (budget mode)",
                    n_changed, filename_only_threshold)

    prev_plan_json = None
    if prev_plan_path and os.path.isfile(prev_plan_path):
        with open(prev_plan_path, "r", encoding="utf-8") as f:
            prev_plan_json = f.read()

    prompt = build_propose_prompt(contract_text, repo_root, git_context,
                                  prev_plan_json, feedback,
                                  filename_only=filename_only,
                                  n_changed=n_changed,
                                  threshold=filename_only_threshold)

    logger.info("Running commit author (single, not fan-out)...")
    logger.debug("Prompt length: %d chars", len(prompt))

    wr = call_worker(provider, model, prompt, cwd=repo_root, timeout=600,
                     **(provider_kwargs or {}))
    if ledger is not None:
        ledger.record_call("commit", "commit-plan", provider, model,
                           prompt=prompt, output=wr.stdout, latency_s=wr.latency_s,
                           ok=wr.exit_code == 0,
                           err=wr.stderr[:200] if wr.exit_code != 0 else "",
                           real_tokens=wr.real_tokens)

    plan = extract_first_json(wr.stdout)  # raises ValueError if no JSON found
    plan = _normalize_plan(plan, repo_root)

    problems = _validate_plan(plan)
    if problems:
        logger.warning("commit-plan: plan has structural problems: %s",
                       "; ".join(problems))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    n_commits = len(plan.get("commits") or [])
    logger.info("Commit-plan written to %s (%d commits, termination=%s)",
                output_path, n_commits, plan.get("termination", "?"))
    return plan


# ── execute (deterministic) ──────────────────────────────────────────────────

def _commit_files(commit: dict[str, Any]) -> list[str]:
    """Return a commit's file list as normalized rel paths (deduped, order kept)."""
    files: list[str] = []
    for f in commit.get("files") or []:
        rel = str(f).strip().replace("\\", "/")
        if rel and rel not in files:
            files.append(rel)
    return files


def build_commit_proposal(plan: dict[str, Any], repo_root: str) -> dict[str, Any]:
    """Evaluate every planned commit against live git state; decide the verdict.

    The proposal is ``ready`` only when ALL of these hold:
      - termination == "ready_to_commit"
      - there is at least one commit
      - every commit is COMMITTABLE: non-empty file list, valid conventional
        message, every file currently changed in git, and no file assigned to
        more than one commit across the plan.

    ``leftover`` (computed) = changed files not covered by any commit — surfaced,
    not blocking (a plan may intentionally leave some files uncommitted).
    """
    termination = plan.get("termination")
    commits = plan.get("commits") if isinstance(plan.get("commits"), list) else []
    gate = plan.get("gate") if isinstance(plan.get("gate"), dict) else {}

    changed = changed_paths(repo_root)
    staged = staged_paths(repo_root)

    # Detect files claimed by more than one commit (hunk-split / double assignment).
    seen: dict[str, int] = {}
    for idx, c in enumerate(commits):
        if not isinstance(c, dict):
            continue
        for rel in _commit_files(c):
            seen[rel] = seen.get(rel, 0) + 1
    duplicated = {rel for rel, n in seen.items() if n > 1}

    covered: set[str] = set()
    commit_results: list[dict[str, Any]] = []
    for idx, c in enumerate(commits):
        if not isinstance(c, dict):
            continue
        cid = c.get("id", f"c{idx + 1}")
        message = c.get("message", "")
        files = _commit_files(c)
        covered.update(files)

        messages: list[str] = []
        status = COMMITTABLE
        if not files:
            status = EMPTY_FILES
            messages.append("commit has no files")
        elif not _MESSAGE_RE.match(message or ""):
            status = BAD_MESSAGE
            messages.append(
                f"message is not 'type(scope): description' with an allowed type "
                f"{_ALLOWED_TYPES}: {message!r}")

        not_changed = [f for f in files if f not in changed]
        dups = [f for f in files if f in duplicated]
        if status == COMMITTABLE and not_changed:
            status = FILE_NOT_CHANGED
            messages.append("not changed in live git (drift/clean/typo): "
                            + ", ".join(not_changed))
        if status == COMMITTABLE and dups:
            status = FILE_DUPLICATED
            messages.append("file(s) assigned to more than one commit: "
                            + ", ".join(dups))

        commit_results.append({
            "id": cid,
            "message": message,
            "files": files,
            "note": c.get("note", ""),
            "status": status,
            "committable": status == COMMITTABLE,
            "messages": messages,
        })

    leftover_set = changed - covered
    leftover = sorted(leftover_set)
    # Staged paths the plan left out are NOT a blocker: staging is the operator's
    # explicit "commit this" intent (§0), and the autonomous pipeline has no human
    # standing by to resolve a halt. They are surfaced here and swept into a final
    # commit by execute_commits (so they are honored, never silently reverted by the
    # per-commit index reset, and never cause a dead stop).
    staged_leftover = sorted(leftover_set & staged)

    reasons: list[str] = []
    if termination != "ready_to_commit":
        reasons.append(f"termination is {termination!r}, not 'ready_to_commit'")
    if not commit_results:
        reasons.append("plan contains no commits (nothing to commit)")
    for r in commit_results:
        if not r["committable"]:
            reasons.append(f"{r['id']}: {r['status']}")
    if gate.get("commit") is True:
        reasons.append("gate.commit was True — ignored (write is human-held)")

    ready = (
        termination == "ready_to_commit"
        and bool(commit_results)
        and all(r["committable"] for r in commit_results)
    )

    return {
        "ready": ready,
        "not_ready_reasons": reasons,
        "repo_root": os.path.abspath(repo_root),
        "termination": termination,
        "commits": commit_results,
        "leftover": leftover,
        "staged_leftover": staged_leftover,
        "declared_leftover": plan.get("leftover") or [],
        "n_commits": len(commit_results),
        "n_committable": sum(1 for r in commit_results if r["committable"]),
    }


def _stage_commit_paths(repo_root: str, files: list[str]) -> tuple[bool, str]:
    """Stage exactly ``files`` into the index — additions, modifications, AND deletions.

    ``git add`` cannot stage a path that is gitignored-but-present on disk ("paths are
    ignored") and fatals on a pathspec matching nothing in the worktree ("did not
    match any files"). Both happen when a commit removes a tracked file that has become
    gitignored (e.g. a regenerated ``__pycache__`` file that was ``git rm --cached``'d).

    So split the paths: a path that is absent OR currently gitignored is a REMOVAL —
    stage it with ``git rm --cached`` (drops it from the index, keeps the on-disk file);
    everything else is an add/modify staged with ``git add``. Scope is preserved (only
    these pathspecs are touched).
    """
    removals: list[str] = []
    adds: list[str] = []
    for rel in files:
        absent = not os.path.exists(os.path.join(repo_root, rel))
        # --no-index: report a match on the .gitignore RULES alone, so a tracked file
        # that now matches an ignore rule (the thing we want to untrack) is detected as
        # a removal. Plain check-ignore never flags a tracked path, which would misroute
        # it to `git add` and re-track it.
        ignored = (not absent) and _git(
            repo_root, ["check-ignore", "--no-index", "-q", "--", rel]).returncode == 0
        (removals if (absent or ignored) else adds).append(rel)

    if adds:
        r = _git(repo_root, ["add", "--", *adds])
        if r.returncode != 0:
            return False, f"git add failed — {r.stderr.strip()}"
    if removals:
        r = _git(repo_root, ["rm", "-r", "--cached", "--ignore-unmatch", "--", *removals])
        if r.returncode != 0:
            return False, f"git rm --cached failed — {r.stderr.strip()}"
    return True, ""


def execute_commits(plan: dict[str, Any], repo_root: str) -> dict[str, Any]:
    """Run ``git add``/``git commit`` for every commit of a READY plan, scoped.

    Precondition: the caller has confirmed ``build_commit_proposal(...)['ready']``.
    Each commit is scoped to its own pathspecs so out-of-scope changes never leak:

      git add    -- <files>     stage exactly this commit's paths (incl. new files)
      git commit -- <files> -m  commit ONLY those paths (other staged paths ignored)

    The sequence is all-or-nothing: HEAD is recorded first; each planned file is
    re-checked as still-changed at the instant of writing; if any git step fails or
    a file has drifted clean, every commit created in this run is undone with
    ``git reset --soft <orig_head>`` (working-tree changes are preserved).

    Returns: ``ok``, ``attempted``, ``committed`` (list of {id, hash, message}),
    ``rolled_back``, ``reason`` on failure.
    """
    result: dict[str, Any] = {
        "ok": False, "attempted": True, "committed": [],
        "rolled_back": False, "reason": "",
    }

    commits = [c for c in (plan.get("commits") or []) if isinstance(c, dict)]
    if not commits:
        result["reason"] = "plan has no commits"
        return result

    orig_head = _head_commit(repo_root)

    # Staging is explicit "commit this" intent (§0). Capture any staged path the
    # plan did not assign BEFORE the per-commit index resets wipe the staging — we
    # commit these in a final sweep rather than halt or silently revert them.
    covered: set[str] = set()
    for c in commits:
        covered.update(_commit_files(c))
    orphan_staged = sorted(staged_paths(repo_root) - covered)

    def _rollback() -> None:
        if orig_head is None:
            # Unborn branch: cannot soft-reset to a sha. Leave commits in place but
            # flag it loudly — extremely rare (first-ever commit to an empty repo).
            logger.error("rollback skipped: repo had no commits before this run; "
                         "remove the created commit(s) manually if needed")
            return
        rb = _git(repo_root, ["reset", "--soft", orig_head])
        result["rolled_back"] = rb.returncode == 0
        if rb.returncode != 0:
            logger.error("rollback (reset --soft %s) failed: %s",
                         orig_head, rb.stderr.strip())

    # Re-verify drift once up front (cheap) so we don't create a partial chain.
    changed = changed_paths(repo_root)
    for c in commits:
        for rel in _commit_files(c):
            if rel not in changed:
                result["reason"] = (
                    f"{c.get('id', '?')}: {rel} is no longer changed in git "
                    "(drift at write time) — nothing committed")
                return result

    committed: list[dict[str, str]] = []
    for c in commits:
        cid = c.get("id", "?")
        message = c.get("message", "")
        files = _commit_files(c)

        # Clean the index so this commit captures EXACTLY its own paths (scoping),
        # stage just this commit's changes, then commit the INDEX (no pathspec).
        # Committing the index — not `git commit -- <paths>` — is what makes a DELETION
        # of a still-present, now-ignored file actually land: a pathspec commit re-reads
        # those paths from the worktree and would keep the file.
        reset = _git(repo_root, ["reset", "-q"])
        if reset.returncode != 0:
            result["reason"] = f"{cid}: git reset (unstage) failed — {reset.stderr.strip()}"
            _rollback()
            return result

        ok, err = _stage_commit_paths(repo_root, files)
        if not ok:
            result["reason"] = f"{cid}: {err}"
            _rollback()
            return result

        com = _git(repo_root, ["commit", "-m", message])
        if com.returncode != 0:
            result["reason"] = f"{cid}: git commit failed — {com.stderr.strip()}"
            _rollback()
            return result

        sha = _head_commit(repo_root) or "?"
        committed.append({"id": cid, "hash": sha, "message": message})
        logger.info("Committed %s %s — %s", cid, sha[:9], message)

    # Sweep any staged path the plan left unassigned into one final commit —
    # mechanical (no grouping judgment); staging already declared the intent.
    if orphan_staged:
        reset = _git(repo_root, ["reset", "-q"])
        if reset.returncode != 0:
            result["reason"] = f"staged-sweep: git reset failed — {reset.stderr.strip()}"
            _rollback()
            return result
        ok, err = _stage_commit_paths(repo_root, orphan_staged)
        if not ok:
            result["reason"] = f"staged-sweep: {err}"
            _rollback()
            return result
        com = _git(repo_root, ["commit", "-m", _SWEEP_MESSAGE])
        if com.returncode != 0:
            result["reason"] = f"staged-sweep: git commit failed — {com.stderr.strip()}"
            _rollback()
            return result
        sha = _head_commit(repo_root) or "?"
        committed.append({"id": "staged-sweep", "hash": sha, "message": _SWEEP_MESSAGE})
        logger.info("Committed staged-sweep %s — %s", sha[:9], _SWEEP_MESSAGE)

    result["ok"] = True
    result["committed"] = committed
    return result


def render_commit_summary_lines(
    proposal: dict[str, Any], max_files: int = 8
) -> list[str]:
    """Render a compact, log-friendly summary of what each commit contains.

    Unlike the full markdown proposal (written only with ``--out``), this is meant for
    stdout/log: per commit it shows the verdict flag, message, file count, and up to
    ``max_files`` paths (the rest collapsed to ``… +N more``) so a thousand-file
    change set stays readable. Leftover/staged-leftover counts are summarized too.
    """
    lines: list[str] = []
    commits = proposal.get("commits") or []
    lines.append(f"Proposed commits ({len(commits)}):")
    if not commits:
        lines.append("  (none)")
    for i, r in enumerate(commits, 1):
        flag = "OK " if r.get("committable") else "BAD"
        files = r.get("files") or []
        shown = ", ".join(files[:max_files])
        if len(files) > max_files:
            shown += f", … +{len(files) - max_files} more"
        lines.append(f"  [{flag}] {i}. {r.get('message', '')}  "
                     f"({len(files)} file{'s' if len(files) != 1 else ''})")
        if shown:
            lines.append(f"        {shown}")
        for msg in r.get("messages") or []:
            lines.append(f"        ! {msg}")
    staged_left = proposal.get("staged_leftover") or []
    if staged_left:
        lines.append(f"Staged but unassigned (auto-committed on --write): "
                     f"{len(staged_left)}")
    leftover = proposal.get("leftover") or []
    if leftover:
        shown = ", ".join(leftover[:max_files])
        if len(leftover) > max_files:
            shown += f", … +{len(leftover) - max_files} more"
        lines.append(f"Uncommitted leftover ({len(leftover)}): {shown}")
    return lines


def render_commit_proposal_markdown(proposal: dict[str, Any]) -> str:
    """Render the human-facing commit proposal (table + verdict + result)."""
    lines: list[str] = []
    write = proposal.get("write")
    if write and write.get("ok"):
        lines.append("# Commit result — COMMITTED to the repo")
    elif write and write.get("attempted"):
        lines.append("# Commit result — commit attempted but NOT completed")
    else:
        lines.append("# Commit proposal — dry run (nothing was committed)")
    lines.append("")
    lines.append(f"- repo_root: `{proposal['repo_root']}`")
    lines.append(f"- plan termination: `{proposal['termination']}`")
    lines.append(f"- commits: {proposal['n_commits']} "
                 f"({proposal['n_committable']} committable)")
    lines.append(f"- uncommitted leftover: {len(proposal['leftover'])}")
    lines.append("")

    if proposal["ready"]:
        lines.append("## ✅ VERDICT: READY TO COMMIT")
        lines.append("")
        lines.append("Every commit's files were re-verified changed in live git. "
                     "Approve by re-running with `--write`.")
    else:
        lines.append("## ⛔ VERDICT: NOT READY")
        lines.append("")
        lines.append("Do not commit. Reasons:")
        for reason in proposal["not_ready_reasons"]:
            lines.append(f"- {reason}")
    lines.append("")

    lines.append("## Proposed commits")
    lines.append("")
    if not proposal["commits"]:
        lines.append("_(none)_")
        lines.append("")
    else:
        lines.append("| # | commit message | files | note |")
        lines.append("|---|---|---|---|")
        for i, r in enumerate(proposal["commits"], 1):
            flag = "✔" if r["committable"] else "✗"
            files = ", ".join(f"`{f}`" for f in r["files"]) or "_(none)_"
            note = r.get("note") or ""
            lines.append(f"| {flag} {i} | `{r['message']}` | {files} | {note} |")
        lines.append("")
        for r in proposal["commits"]:
            if r["messages"]:
                lines.append(f"- **{r['id']}** (`{r['status']}`):")
                for msg in r["messages"]:
                    lines.append(f"  - ⚠ {msg}")
        lines.append("")

    if write and write.get("attempted"):
        if write.get("ok"):
            lines.append("## Write — COMMITTED")
            lines.append("")
            for c in write.get("committed", []):
                lines.append(f"- `{c['hash'][:9]}` {c['id']} — {c['message']}")
        else:
            lines.append("## Write — NOT COMPLETED")
            lines.append("")
            lines.append(f"- {write.get('reason', 'commit did not complete')}")
            if write.get("rolled_back"):
                lines.append("- created commits were rolled back "
                             "(`git reset --soft`); working tree preserved")
        lines.append("")

    if proposal.get("staged_leftover"):
        lines.append("## Staged but in no commit — will be auto-committed")
        lines.append("")
        lines.append("These paths are staged (explicit commit intent) yet no commit "
                     "covers them. On `--write` they are swept into a final "
                     f"`{_SWEEP_MESSAGE}` commit, so the staging is honored rather "
                     "than silently reverted by the per-commit index reset.")
        lines.append("")
        for rel in proposal["staged_leftover"]:
            lines.append(f"- `{rel}`")
        lines.append("")

    if proposal["leftover"]:
        lines.append("## Uncommitted leftover (changed but in no commit)")
        lines.append("")
        for rel in proposal["leftover"]:
            lines.append(f"- `{rel}`")
        lines.append("")

    return "\n".join(lines)


def run_commit(
    plan_path: str,
    repo_root: str | None = None,
    output_path: str | None = None,
    write: bool = False,
) -> dict[str, Any]:
    """Run the commit stage: commit-plan JSON + live git → proposal (+ optional write).

    Loads the SSOT plan, re-verifies every commit against the live git state,
    renders a proposal table, and (optionally) executes the commits. Calls no
    worker — fully deterministic. Returns the proposal dict.

    With ``write=False`` (default) nothing is committed. With ``write=True`` AND a
    READY plan, each commit is created scoped to its pathspecs, all-or-nothing.

    Raises:
        FileNotFoundError: if the plan file is missing.
        ValueError: if repo_root cannot be resolved (not given and absent from plan).
    """
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    root = repo_root or plan.get("repo_root")
    if not root:
        raise ValueError("repo_root not provided and not present in the commit-plan")
    if _git(root, ["rev-parse", "--is-inside-work-tree"]).returncode != 0:
        raise ValueError(f"not a git work tree: {root}")

    proposal = build_commit_proposal(plan, root)

    if write:
        if proposal["ready"]:
            proposal["write"] = execute_commits(plan, root)
        else:
            logger.warning("commit: --write requested but plan is NOT READY — "
                           "nothing committed")
            proposal["write"] = {
                "ok": False, "attempted": False, "committed": [],
                "rolled_back": False,
                "reason": "plan not ready — refusing to commit",
            }
    else:
        proposal["write"] = None

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(render_commit_proposal_markdown(proposal))
        logger.info("Commit proposal written to %s", output_path)

    verdict = "READY" if proposal["ready"] else "NOT READY"
    logger.info("Commit: %s — %d/%d commits committable, termination=%s",
                verdict, proposal["n_committable"], proposal["n_commits"],
                proposal["termination"])
    if not proposal["ready"]:
        for reason in proposal["not_ready_reasons"]:
            logger.info("  not-ready: %s", reason)

    w = proposal.get("write")
    if w and w.get("ok"):
        logger.info("Write: COMMITTED %d commit(s)", len(w.get("committed", [])))
    elif w and w.get("attempted"):
        logger.warning("Write: NOT completed — %s", w.get("reason"))
    return proposal
