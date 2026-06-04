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
    build_repo_tree, build_decompose_prompt, build_literal_preview, _tree_useful,
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


# ── Literal pre-grep (M012): free bait, never a menu ───────────────────────────

def test_literal_preview_maps_seed_tokens_to_real_lines(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "server/db/orders.py",
         "def get_pending(group):\n    return query(order_doc_id, group_head)\n")
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "x"], check=True)

    # The seed NAMES the literal identifier; the grep resolves it to a real line.
    preview = build_literal_preview(
        "the function reading order_doc_id returns the wrong head", repo)
    assert "order_doc_id" in preview
    assert "server/db/orders.py:" in preview


def test_literal_preview_empty_on_pure_symptom_seed(tmp_path):
    # M012's predicted no-op: a symptom-only prose seed carries no literal token,
    # so extract_keywords yields nothing → empty block → caller omits it.
    repo = _git_repo(tmp_path)
    _add(repo, "server/db/orders.py", "x = 1\n")
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "x"], check=True)

    assert build_literal_preview("the ordering does not work right", repo) == ""


def test_literal_preview_empty_when_no_root():
    assert build_literal_preview("order_doc_id is wrong", None) == ""
    assert build_literal_preview("order_doc_id is wrong", "") == ""


def test_literal_preview_drops_flood_keeps_specific(tmp_path):
    # A common token (appears in many files) is too generic to anchor and must be
    # dropped; a specific identifier (one file) must survive. Without this the
    # per-file ripgrep cap let a common token flood the block (450-line bug).
    repo = _git_repo(tmp_path)
    for i in range(30):                       # `widget` floods 30 files
        _add(repo, f"src/mod_{i}.py", "widget = 1\n")
    _add(repo, "src/special.py",
         'launch_sequence_handler("x")  # unique_marker_xyz lives here\n')
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "x"], check=True)

    preview = build_literal_preview(
        'the "unique_marker_xyz" path and widget handling are wrong',
        repo, max_files_per_literal=8)
    assert "unique_marker_xyz" in preview          # specific literal kept
    assert "src/special.py" in preview
    # widget grounded to 30 > 8 files → dropped, so the block stays tight.
    assert "mod_0.py" not in preview
    assert preview.count("\n") < 8                  # no flood


def test_prompt_embeds_literal_preview_with_bait_framing():
    prompt = build_decompose_prompt(
        "fix it", literal_preview="order_doc_id:\n  server/db/orders.py:2  ...")
    assert "LITERAL PRE-GREP" in prompt
    assert "NOT a menu" in prompt          # anti-anchoring framing (M012 §4)
    assert "invent it" in prompt           # rule 6 stays in force
    assert "order_doc_id" in prompt


def test_prompt_omits_literal_section_when_none():
    prompt = build_decompose_prompt("fix it", literal_preview="")
    assert "LITERAL PRE-GREP" not in prompt


def test_contract_carries_coverage_risk_self_doubt_flag():
    # B1: the queen is asked to self-flag axes whose blind grep may find nothing,
    # so a downstream (flag ∧ empty-FIND) check can reinforce only those.
    prompt = build_decompose_prompt("fix it")
    assert "coverage_risk" in prompt
    assert '"thin"' in prompt                  # the canonical flag value
    assert "self-doubt" in prompt              # framed as confidence, not a verdict
