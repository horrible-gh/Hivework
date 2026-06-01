"""Tests for the decompose stage's repo-tree feeding (hive/decompose.py).

The queen used to fan out blind to the repo and guess file_globs, so a frontend
off-by-one whose Vue component the FE axis never located slipped through
(NR168). We now feed a free, local, deterministic file map so axes anchor on
real paths. These cover the pure tree-building + prompt-embedding (no worker
call, no model).
"""
import os
import subprocess

from hive.decompose import (
    build_repo_tree, build_decompose_prompt, _tree_useful,
)


def _git_repo(tmp_path):
    repo = str(tmp_path)
    subprocess.run(["git", "-C", repo, "init", "-q"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.email", "t@t.t"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "t"], check=True)
    return repo


def _add(repo, rel, text="x\n"):
    path = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    subprocess.run(["git", "-C", repo, "add", "--", rel], check=True)


def test_tree_useful_drops_binaries_and_noise_dirs():
    assert _tree_useful("client/src/components/DocWorkflow.vue")
    assert _tree_useful("server/modules/flow_gate/db/workflow_sequences.py")
    assert not _tree_useful("client/assets/logo.png")
    assert not _tree_useful("node_modules/foo/index.js")
    assert not _tree_useful("client/node_modules/dep/x.js")
    assert not _tree_useful("package-lock.lock")


def test_build_repo_tree_lists_tracked_source_files(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "client/src/main/components/DocWorkflow.vue")
    _add(repo, "server/modules/flow_gate/db/workflow_sequences.py")
    _add(repo, "client/assets/logo.png")          # binary → excluded
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "x"], check=True)

    tree = build_repo_tree(repo)
    assert "client/src/main/components/DocWorkflow.vue" in tree
    assert "server/modules/flow_gate/db/workflow_sequences.py" in tree
    assert "logo.png" not in tree


def test_build_repo_tree_collapses_large_repo_to_dir_counts(tmp_path):
    repo = _git_repo(tmp_path)
    for i in range(30):
        _add(repo, f"server/mod/a{i}.py")
    for i in range(30):
        _add(repo, f"client/src/c{i}.vue")
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "x"], check=True)

    tree = build_repo_tree(repo, max_lines=10)  # 60 files > cap → collapse
    assert "file count" in tree.lower() or "files)" in tree
    assert "server/mod/" in tree and "client/src/" in tree


def test_build_repo_tree_empty_for_missing_root():
    assert build_repo_tree(None) == ""
    assert build_repo_tree("") == ""


def test_prompt_embeds_tree_and_instructs_real_paths():
    prompt = build_decompose_prompt(
        "find the off-by-one", repo_tree="client/src/DocWorkflow.vue")
    # the tree-box header ("REALLY exist") and the path are present only when a
    # tree is supplied (the contract body always mentions "REPO FILE TREE").
    assert "REALLY exist" in prompt
    assert "client/src/DocWorkflow.vue" in prompt


def test_prompt_omits_tree_section_when_none():
    prompt = build_decompose_prompt("find the off-by-one", repo_tree="")
    assert "REALLY exist" not in prompt
