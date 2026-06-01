"""Tests for the commit stage (hive/commit.py).

The execute path is deterministic and calls no worker, so these tests need no
provider mocking — they build a real throwaway git repo, author a commit-plan by
hand, and assert on the proposal verdict, the rendered table, and (for --write)
the commits actually created plus all-or-nothing rollback.

``run_propose`` (the author worker) is not exercised here (it needs a live model);
its plan validation and normalization are covered via build_commit_proposal and
_normalize_plan.
"""

import json
import os
import subprocess

import pytest

from hive.commit import (
    build_commit_proposal,
    changed_paths,
    execute_commits,
    render_commit_proposal_markdown,
    run_commit,
    staged_paths,
    _normalize_plan,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def _repo(tmp_path):
    """Init a git repo with one committed baseline file, return its path."""
    repo = str(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "commit", "--allow-empty", "-q", "-m", "chore: baseline")
    return repo


def _write(repo, rel, text):
    path = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return rel


def _plan(commits, termination="ready_to_commit", **extra):
    return {
        "commits": commits,
        "leftover": [],
        "gate": {"commit": False},
        "termination": termination,
        **extra,
    }


def _has_git():
    try:
        subprocess.run(["git", "--version"], capture_output=True)
        return True
    except (OSError, FileNotFoundError):
        return False


pytestmark = pytest.mark.skipif(not _has_git(), reason="git not available")


# ── changed_paths ────────────────────────────────────────────────────────────

def test_changed_paths_sees_modified_and_untracked(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")          # untracked
    changed = changed_paths(repo)
    assert "a.py" in changed


# ── build_commit_proposal ────────────────────────────────────────────────────

def test_ready_when_all_committable(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")
    plan = _plan([{"id": "c1", "message": "feat(core): add a", "files": ["a.py"]}])
    proposal = build_commit_proposal(plan, repo)
    assert proposal["ready"] is True
    assert proposal["n_committable"] == 1


def test_not_ready_on_bad_message(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")
    plan = _plan([{"id": "c1", "message": "added a file", "files": ["a.py"]}])
    proposal = build_commit_proposal(plan, repo)
    assert proposal["ready"] is False
    assert any(r["status"] == "bad_message" for r in proposal["commits"])


def test_not_ready_on_file_in_two_commits(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")
    plan = _plan([
        {"id": "c1", "message": "feat(core): add a", "files": ["a.py"]},
        {"id": "c2", "message": "fix(core): tweak a", "files": ["a.py"]},
    ])
    proposal = build_commit_proposal(plan, repo)
    assert proposal["ready"] is False
    assert any(r["status"] == "file_duplicated" for r in proposal["commits"])


def test_not_ready_on_unchanged_file(tmp_path):
    repo = _repo(tmp_path)
    # b.py is referenced by the plan but never created/changed → drift.
    plan = _plan([{"id": "c1", "message": "feat(core): add b", "files": ["b.py"]}])
    proposal = build_commit_proposal(plan, repo)
    assert proposal["ready"] is False
    assert any(r["status"] == "file_not_changed" for r in proposal["commits"])


def test_leftover_is_computed(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")
    _write(repo, "scratch.txt", "tmp\n")
    plan = _plan([{"id": "c1", "message": "feat(core): add a", "files": ["a.py"]}])
    proposal = build_commit_proposal(plan, repo)
    assert "scratch.txt" in proposal["leftover"]


def _repo_with_staged_untracking(tmp_path):
    """Repo where cache/x.pyc is a STAGED deletion (git rm --cached) still on disk,
    and .gitignore is an untracked change. Mirrors the real __pycache__ case where the
    PM stages an untracking before asking for a commit plan.
    """
    repo = _repo(tmp_path)
    _write(repo, "cache/x.pyc", "y\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "track x.pyc")
    _write(repo, ".gitignore", "cache/\n")          # untracked
    _git(repo, "rm", "--cached", "-q", "cache/x.pyc")  # staged deletion, kept on disk
    assert "cache/x.pyc" in staged_paths(repo)       # precondition
    return repo


def test_staged_path_left_uncommitted_is_surfaced_not_blocked(tmp_path):
    """A staged path the plan drops no longer halts: staging is commit intent, so it
    is surfaced (and swept into a final commit on --write), never a dead stop.
    """
    repo = _repo_with_staged_untracking(tmp_path)
    plan = _plan([{"id": "c1", "message": "chore(git): ignore cache",
                   "files": [".gitignore"]}])
    proposal = build_commit_proposal(plan, repo)
    assert proposal["ready"] is True
    assert "cache/x.pyc" in proposal["staged_leftover"]
    assert not any("staged" in r for r in proposal["not_ready_reasons"])


def test_ready_when_staged_path_is_committed(tmp_path):
    """Including the staged untracking in a commit is READY — the guard does not
    over-block; it only fires on staged paths left out of the plan.
    """
    repo = _repo_with_staged_untracking(tmp_path)
    plan = _plan([{"id": "c1", "message": "chore(git): stop tracking cache",
                   "files": [".gitignore", "cache/x.pyc"]}])
    proposal = build_commit_proposal(plan, repo)
    assert proposal["ready"] is True
    assert proposal["staged_leftover"] == []


def test_write_sweeps_staged_untracking_into_commit(tmp_path):
    """End-to-end: a plan that omits the staged untracking no longer halts — the
    staged `git rm --cached` is swept into a final commit, honoring the staging
    instead of silently reverting it (the original bug) or blocking on it.
    """
    repo = _repo_with_staged_untracking(tmp_path)
    plan = _plan([{"id": "c1", "message": "chore(git): ignore cache",
                   "files": [".gitignore"]}])
    plan_path = os.path.join(repo, "plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f)

    proposal = run_commit(plan_path, repo_root=repo, write=True)
    assert proposal["ready"] is True
    assert proposal["write"]["ok"] is True
    # The untracking was committed: x.pyc is no longer tracked nor staged, and a
    # dedicated sweep commit sits alongside the planned one.
    assert _git(repo, "ls-files", "cache/x.pyc").stdout.strip() == ""
    assert "cache/x.pyc" not in staged_paths(repo)
    ids = [c["id"] for c in proposal["write"]["committed"]]
    assert "c1" in ids and "staged-sweep" in ids


def test_not_ready_when_termination_needs_pm(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")
    plan = _plan([{"id": "c1", "message": "feat(core): add a", "files": ["a.py"]}],
                 termination="needs_pm")
    proposal = build_commit_proposal(plan, repo)
    assert proposal["ready"] is False


# ── normalize ────────────────────────────────────────────────────────────────

def test_normalize_forces_gate_commit_false(tmp_path):
    plan = {"commits": [], "gate": {"commit": True}, "termination": "ready_to_commit"}
    out = _normalize_plan(plan, str(tmp_path))
    assert out["gate"]["commit"] is False
    assert out["repo_root"] == os.path.abspath(str(tmp_path))


# ── execute_commits (the actual git mutation) ────────────────────────────────

def test_execute_creates_scoped_commits(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")
    _write(repo, "README.md", "# docs\n")
    plan = _plan([
        {"id": "c1", "message": "feat(core): add a", "files": ["a.py"]},
        {"id": "c2", "message": "docs(readme): add readme", "files": ["README.md"]},
    ])
    result = execute_commits(plan, repo)
    assert result["ok"] is True
    assert len(result["committed"]) == 2

    log = _git(repo, "log", "--oneline").stdout
    assert "feat(core): add a" in log
    assert "docs(readme): add readme" in log
    # Each commit is scoped: c1 touched only a.py.
    files_c1 = _git(repo, "show", "--name-only", "--format=",
                    result["committed"][0]["hash"]).stdout
    assert "a.py" in files_c1
    assert "README.md" not in files_c1
    # Working tree is clean afterwards (both files committed).
    assert changed_paths(repo) == set()


def test_run_commit_dry_run_does_not_commit(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")
    plan = _plan([{"id": "c1", "message": "feat(core): add a", "files": ["a.py"]}])
    plan_path = os.path.join(repo, "plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f)

    before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    proposal = run_commit(plan_path, repo_root=repo, write=False)
    after = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert proposal["ready"] is True
    assert before == after  # nothing committed
    assert proposal["write"] is None


def test_run_commit_write_creates_commit(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")
    plan = _plan([{"id": "c1", "message": "feat(core): add a", "files": ["a.py"]}])
    plan_path = os.path.join(repo, "plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f)

    proposal = run_commit(plan_path, repo_root=repo, write=True)
    assert proposal["write"]["ok"] is True
    log = _git(repo, "log", "--oneline").stdout
    assert "feat(core): add a" in log


def test_write_refused_when_not_ready(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")
    plan = _plan([{"id": "c1", "message": "bad message", "files": ["a.py"]}])
    plan_path = os.path.join(repo, "plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f)

    before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    proposal = run_commit(plan_path, repo_root=repo, write=True)
    after = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert proposal["ready"] is False
    assert proposal["write"]["attempted"] is False
    assert before == after  # not-ready plan never commits


def test_execute_rolls_back_on_midsequence_failure(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")
    orig = _git(repo, "rev-parse", "HEAD").stdout.strip()
    # c1 is valid; c2 names a file that isn't changed → up-front drift check trips
    # before ANY commit is made. (Belt-and-suspenders: nothing partial.)
    plan = _plan([
        {"id": "c1", "message": "feat(core): add a", "files": ["a.py"]},
        {"id": "c2", "message": "feat(core): add ghost", "files": ["ghost.py"]},
    ])
    result = execute_commits(plan, repo)
    assert result["ok"] is False
    after = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert after == orig  # no partial commit chain left behind


def test_render_markdown_smoke(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.py", "x = 1\n")
    plan = _plan([{"id": "c1", "message": "feat(core): add a", "files": ["a.py"]}])
    md = render_commit_proposal_markdown(build_commit_proposal(plan, repo))
    assert "VERDICT" in md
    assert "feat(core): add a" in md


def test_write_commits_a_deletion(tmp_path):
    """A planned commit that DELETES a file (e.g. a now-ignored, git-rm'd path) must
    stage via ``git add -A`` and commit cleanly. Plain ``git add -- <path>`` fatals on
    a deleted pathspec ("did not match any files"), which previously broke any commit
    that included a removal.
    """
    repo = _repo(tmp_path)
    _write(repo, "junk.pyc", "x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add junk")
    os.remove(os.path.join(repo, "junk.pyc"))  # worktree deletion to be committed

    plan = _plan([{"id": "c1", "message": "chore(git): remove junk.pyc",
                   "files": ["junk.pyc"]}])
    result = execute_commits(plan, repo)

    assert result["ok"] is True, result.get("reason")
    show = _git(repo, "show", "--stat", "--oneline", "HEAD")
    assert "junk.pyc" in show.stdout          # the deletion landed in the commit
    assert not os.path.exists(os.path.join(repo, "junk.pyc"))


def test_write_untracks_ignored_present_file(tmp_path):
    """The real __pycache__ case: a tracked file becomes gitignored but is STILL on
    disk (regenerated by a tool). The commit must untrack it (git rm --cached) and keep
    the on-disk file — plain 'git add' fatals here with 'paths are ignored'.
    """
    repo = _repo(tmp_path)
    _write(repo, "cache/x.pyc", "y\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "track x.pyc")
    # Ignore the dir, stage the untracking, but keep the file on disk (regenerated).
    _write(repo, ".gitignore", "cache/\n")
    _git(repo, "rm", "--cached", "-q", "cache/x.pyc")
    assert os.path.exists(os.path.join(repo, "cache/x.pyc"))  # precondition: still on disk

    plan = _plan([{"id": "c1", "message": "chore(git): stop tracking cache",
                   "files": [".gitignore", "cache/x.pyc"]}])
    result = execute_commits(plan, repo)

    assert result["ok"] is True, result.get("reason")
    assert _git(repo, "ls-files", "cache/x.pyc").stdout.strip() == ""   # untracked now
    assert os.path.exists(os.path.join(repo, "cache/x.pyc"))            # kept on disk
