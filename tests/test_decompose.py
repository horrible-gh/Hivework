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
    FE_DERIVED_AXIS_ID, build_repo_tree, build_decompose_prompt,
    build_literal_preview, ensure_fe_derived_state_axis,
    is_fe_derived_state_symptom, _frontend_source_globs, _tree_useful,
    independent_axes,
)
from hive.retriever import retrieve
from hive.searchplan import task_to_searchplan


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


def test_decompose_retries_past_clean_blank_comb_then_succeeds(tmp_path, monkeypatch):
    # T905: codex finished CLEAN (rc=0, no stderr, well under timeout) yet returned a
    # BLANK final message. With retries>0 the queen call is re-fired and the run survives.
    from hive import decompose as dec
    from hive.providers import WorkerResult
    monkeypatch.chdir(tmp_path)  # decompose_raw_last.txt is written to cwd
    calls = {"n": 0}

    def fake(provider, model, prompt, cwd=None, timeout=300, on_start=None, **kw):
        calls["n"] += 1
        if calls["n"] <= 2:  # first two attempts: the clean rc=0 blank that bit T905
            return WorkerResult(stdout="\n", stderr="", exit_code=0, latency_s=24.1)
        return WorkerResult(stdout='{"tasks": [{"id": "A"}]}', stderr="",
                            exit_code=0, latency_s=12.0)

    monkeypatch.setattr(dec, "call_worker", fake)
    out = dec.run_decompose(seed_text="boom", retries=2)
    assert calls["n"] == 3  # blank, blank, good
    assert out["tasks"][0]["id"] == "A"


def test_decompose_blank_exhausts_retries_with_honest_error(tmp_path, monkeypatch):
    # When every attempt blanks, the error reports the ACTUAL signals (rc/latency/
    # out_chars) — not the old misleading "(quota/timeout/rc!=0?)" guess that sent
    # T905 debugging toward zombie-locks and quota when the queen just answered blank.
    import pytest
    from hive import decompose as dec
    from hive.providers import WorkerResult
    monkeypatch.chdir(tmp_path)
    calls = {"n": 0}

    def fake(provider, model, prompt, cwd=None, timeout=300, on_start=None, **kw):
        calls["n"] += 1
        return WorkerResult(stdout="", stderr="", exit_code=0, latency_s=24.1)

    monkeypatch.setattr(dec, "call_worker", fake)
    with pytest.raises(ValueError) as ei:
        dec.run_decompose(seed_text="boom", retries=2)
    assert calls["n"] == 3  # 1 + 2 retries, all blank
    msg = str(ei.value)
    assert "rc=0" in msg and "out_chars=0" in msg  # honest, observed signals
    assert "quota/timeout/rc!=0?" not in msg       # the guess is gone


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


# ── Conditional FE derived-state axis (M035) ──────────────────────────────────

M035_SEED = (
    "The workflow head is shifted by one step: the wrong current stage is "
    "highlighted and the done/current/future colors are off by one."
)
M036_SEED = (
    "The HTTP data source returns an empty workflow_head_type field, so inspect "
    "the API response serializer and database producer."
)


def _backend_only_decomposition():
    return {
        "fanout_decision": "fanout",
        "reason": "backend hypotheses",
        "steps": [["SQL_HEAD"]],
        "tasks": [{
            "id": "SQL_HEAD",
            "title": "SQL head ordering",
            "brief": "Inspect get_effective_head ORDER BY and service callers.",
            "depends_on": [],
            "coverage_risk": "ok",
            "search_plan": {
                "keywords": ["get_effective_head", "ORDER BY"],
                "file_globs": ["server/**/*.py", "server/sql/**/*.json"],
                "doc_topics": [],
            },
        }],
    }


def _make_flowgate_fe(repo):
    _add(
        repo,
        "client/src/main/workflow/workflowViewState.ts",
        "type StepVisual = 'done' | 'current' | 'future'\n"
        "export function buildStepStates(workflowSteps: string[], headType: string) {\n"
        "  const headIndex = workflowSteps.indexOf(headType)\n"
        "  return workflowSteps.map((_, idx) =>\n"
        "    idx < headIndex ? 'done' : idx === headIndex ? 'current' : 'future')\n"
        "}\n",
    )
    _add(
        repo,
        "client/src/main/components/DocWorkflow.vue",
        "<template><div v-for=\"step in stepStates\" :class=\"step\" /></template>\n",
    )
    _add(
        repo,
        "server/sql/queries/queries.json",
        '{"get_effective_head":"SELECT * FROM wsi ORDER BY sort_order ASC"}\n',
    )
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "fixture"], check=True)


def test_m035_symptom_injects_fe_axis_but_m036_http_does_not(tmp_path):
    repo = _git_repo(tmp_path)
    _make_flowgate_fe(repo)

    m035 = _backend_only_decomposition()
    ensure_fe_derived_state_axis(m035, M035_SEED, repo)
    assert m035["tasks"][0]["id"] == FE_DERIVED_AXIS_ID
    assert m035["steps"][0][0] == FE_DERIVED_AXIS_ID
    assert len(m035["tasks"]) == 2

    m036 = _backend_only_decomposition()
    ensure_fe_derived_state_axis(m036, M036_SEED, repo)
    assert [task["id"] for task in m036["tasks"]] == ["SQL_HEAD"]


def test_fe_axis_reuses_existing_axis_without_increasing_paid_axis_count(tmp_path):
    repo = _git_repo(tmp_path)
    _make_flowgate_fe(repo)
    result = _backend_only_decomposition()
    result["tasks"].insert(0, {
        "id": "FE_VIEW",
        "title": "Frontend view-state derivation",
        "brief": "Inspect current step state and its Vue renderer.",
        "depends_on": [],
        "search_plan": {
            "keywords": ["current"],
            "file_globs": ["client/src/**/*.vue"],
            "doc_topics": [],
        },
    })
    before = len(result["tasks"])

    ensure_fe_derived_state_axis(result, M035_SEED, repo)

    assert len(result["tasks"]) == before
    fe = result["tasks"][0]
    assert "headIndex" in fe["search_plan"]["keywords"]
    assert (
        "client/src/main/workflow/workflowViewState.ts"
        in fe["search_plan"]["file_globs"]
    )


def test_m035_fe_axis_retrieve_reaches_workflow_view_state(tmp_path):
    repo = _git_repo(tmp_path)
    _make_flowgate_fe(repo)
    result = _backend_only_decomposition()
    ensure_fe_derived_state_axis(result, M035_SEED, repo)
    fe_task = result["tasks"][0]

    plan = task_to_searchplan(fe_task)
    bundle = retrieve(plan, repo, k=3, top_files=8, blame_files=0)

    candidates = {
        item["file"].removeprefix("./") for item in
        bundle["code_snippets"] + bundle["call_chain"]
    }
    assert "client/src/main/workflow/workflowViewState.ts" in candidates
    assert bundle["stats"]["snippets"] > 0


def test_fe_axis_without_frontend_scope_stays_honestly_thin(tmp_path):
    repo = _git_repo(tmp_path)
    _add(repo, "server/workflow.py",
         "currentStep = 'backend-only name must not become an FE candidate'\n")
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "backend"], check=True)
    result = _backend_only_decomposition()
    ensure_fe_derived_state_axis(result, M035_SEED, repo)
    fe_task = result["tasks"][0]

    assert fe_task["search_plan"]["file_globs"] == []
    assert fe_task["search_plan"]["keywords"] == []
    bundle = retrieve(task_to_searchplan(fe_task), repo, blame_files=0)
    assert bundle["code_snippets"] == []
    assert bundle["call_chain"] == []


def test_fe_symptom_gate_is_conservative():
    assert is_fe_derived_state_symptom(M035_SEED)
    assert is_fe_derived_state_symptom(
        "The current step highlight color is wrong in the workflow bar.")
    assert is_fe_derived_state_symptom(
        "The workflow view-state derivation is stale.")
    assert not is_fe_derived_state_symptom(M036_SEED)
    assert not is_fe_derived_state_symptom(
        "The API returns done/current/future status values in its JSON payload.")


def test_frontend_globs_are_derived_only_from_existing_frontend_roots(tmp_path):
    repo = _git_repo(tmp_path)
    _make_flowgate_fe(repo)
    assert _frontend_source_globs(repo) == [
        "client/src/main/workflow/workflowViewState.ts",
        "client/src/main/components/DocWorkflow.vue",
    ]


# ── RC-B: only independent axes fan out to the swarm (NR 0008.0009) ────────────

def test_independent_axes_drops_dependent_synthesis():
    # The run-451 shape: 9 independent leaves + 1 synthesis task that depends on
    # all of them. The synthesis task is NOT swarmable (a blind drone can't see the
    # other combs) and must be excluded from the fan-out.
    tasks = [
        {"id": "A", "depends_on": []},
        {"id": "B"},                       # absent depends_on == independent
        {"id": "synthesis", "depends_on": ["A", "B"]},
    ]
    kept = independent_axes(tasks)
    assert [t["id"] for t in kept] == ["A", "B"]


def test_independent_axes_keeps_all_when_none_dependent():
    tasks = [{"id": "A"}, {"id": "B", "depends_on": []}]
    assert independent_axes(tasks) == tasks


def test_independent_axes_preserves_order():
    tasks = [{"id": "C"}, {"id": "syn", "depends_on": ["C"]}, {"id": "A"}, {"id": "B"}]
    assert [t["id"] for t in independent_axes(tasks)] == ["C", "A", "B"]
