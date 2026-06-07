"""Tests for hive.retriever glob normalization.

Regression guard for the investigate_e2e run-1 silent-empty bug: the queen emits
ABSOLUTE-path globs, but ``rg -g`` matches its pattern against paths RELATIVE to
the search root, so an absolute glob excluded everything → 0 hits → the JUDGE
ruled on an empty bundle. ``_norm_glob`` relativizes (or basename-degrades) each
glob so ``rg -g`` can actually match. These were invisible to the stub suite
because stubs never exercised real ``rg`` glob semantics on absolute paths.
"""
import os

from hive.retriever import (
    _norm_glob, _validate_globs, _partition_globs, _abs_under, _widen_globs,
    _looks_like_label_ref, _extract_value, retrieve, SearchPlan,
    _path_segs, _route_suffix_match, _resolve_http_bindings, _read_def_body,
    _resolve_peer_patterns, _stacking_profile,
    _harvest_inscope_fetch_urls, _covered_ranges, _follow_calls,
    _resolve_field_producers, _resolve_http_producer_paths,
)

ROOT = "C:/workspace/projects/Documents/projects/FlowGate"


def test_absolute_glob_under_root_becomes_relative():
    g = "C:/workspace/projects/Documents/projects/FlowGate/210_design/D031_*.md"
    assert _norm_glob(g, ROOT) == "210_design/D031_*.md"


def test_absolute_glob_with_backslashes_normalizes_to_relative():
    g = r"C:\workspace\projects\Documents\projects\FlowGate\420_task_reports\TR880_*.md"
    assert _norm_glob(g, ROOT) == "420_task_reports/TR880_*.md"


def test_case_insensitive_drive_prefix_match():
    # Windows paths: queen may differ in drive-letter / separator case.
    g = "c:/WORKSPACE/projects/Documents/projects/FlowGate/210_design/**/*.md"
    assert _norm_glob(g, ROOT) == "210_design/**/*.md"


def test_absolute_glob_outside_root_degrades_to_basename():
    # Code repo lives elsewhere; an out-of-root absolute glob can't be relativized,
    # so it degrades to a basename pattern rg matches at any depth (better than the
    # old behaviour of matching nothing).
    g = "C:/workspace/projects/FlowGate/client/src/**/*.vue"
    assert _norm_glob(g, ROOT) == "*.vue"


def test_relative_glob_passes_through():
    assert _norm_glob("210_design/**/*.md", ROOT) == "210_design/**/*.md"


def test_basename_only_glob_passes_through():
    assert _norm_glob("*.md", ROOT) == "*.md"


def test_root_with_trailing_slash_is_handled():
    g = "C:/workspace/projects/Documents/projects/FlowGate/210_design/D031_*.md"
    assert _norm_glob(g, ROOT + "/") == "210_design/D031_*.md"


# ── Doubled-separator collapse: the queen over-escapes a Windows path
#    (``C:\\\\…`` → parsed ``C:\\…``), which ``\\``→``/`` turns into ``C://…`` — a
#    doubled-slash glob rg matches against NOTHING (T890: D031 never retrieved).

def test_doubled_backslash_glob_relativized():
    # Parsed value of an over-escaped queen glob: every separator is two backslashes.
    g = "C:\\\\workspace\\\\projects\\\\Documents\\\\projects\\\\FlowGate\\\\210_design\\\\D031_*.md"
    assert _norm_glob(g, ROOT) == "210_design/D031_*.md"


def test_doubled_forward_slash_glob_collapsed_and_relativized():
    g = "C://workspace//projects//Documents//projects//FlowGate//210_design//**//D031_*.md"
    assert _norm_glob(g, ROOT) == "210_design/**/D031_*.md"


def test_doubled_slash_outside_root_degrades_to_basename():
    g = "C://other//place//D031_*.md"
    assert _norm_glob(g, ROOT) == "D031_*.md"


# ── _validate_globs: existence-probe defence against garbage / over-broad globs.
#    These run real ``rg --files`` against a temp tree, the only thing that can
#    distinguish a structurally-valid non-path (``message/author/date``) from a
#    real one (``server/sql``) — exactly the gap a text filter cannot close.

def _make_tree(tmp_path):
    """A small repo: server/sql/q.json, server/mod/a.py..., plus many docs."""
    (tmp_path / "server" / "sql").mkdir(parents=True)
    (tmp_path / "server" / "sql" / "q.json").write_text("{}", encoding="utf-8")
    (tmp_path / "server" / "mod").mkdir(parents=True)
    for i in range(5):
        (tmp_path / "server" / "mod" / f"a{i}.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    for i in range(40):
        (tmp_path / "docs" / f"d{i}.md").write_text("# doc\n", encoding="utf-8")
    return str(tmp_path)


def test_garbage_glob_zero_match_is_dropped(tmp_path):
    # git-log fields the queen mistook for a path → 0 files → dropped.
    root = _make_tree(tmp_path)
    kept, diag = _validate_globs(
        ["server/sql/*.json", "message/author/date/**/*"], root)
    assert kept == ["server/sql/*.json"]
    assert "message/author/date/**/*" in diag["dropped_empty"]


def test_overbroad_glob_dropped_when_narrower_exists(tmp_path):
    # **/* matches the whole tree; a narrow usable glob is present → drop the broad one.
    root = _make_tree(tmp_path)
    kept, diag = _validate_globs(
        ["server/sql/*.json", "**/*"], root, overbroad_files=10)
    assert kept == ["server/sql/*.json"]
    assert "**/*" in diag["dropped_overbroad"]


def test_overbroad_kept_as_last_resort(tmp_path):
    # No narrower scope survives → keep the broad glob rather than search nothing.
    root = _make_tree(tmp_path)
    kept, diag = _validate_globs(["**/*"], root, overbroad_files=10)
    assert kept == ["**/*"]
    assert diag["dropped_overbroad"] == []  # not reported as dropped when kept


def test_all_garbage_falls_back_to_whole_tree(tmp_path):
    # Every glob is garbage → kept=[] so rg searches the whole tree (a grounded
    # hit anywhere beats a guaranteed-empty bundle on nonexistent dirs).
    root = _make_tree(tmp_path)
    kept, diag = _validate_globs(["no/such/dir/**/*", "also/missing/**/*"], root)
    assert kept == []
    assert len(diag["dropped_empty"]) == 2


def test_usable_globs_pass_through(tmp_path):
    root = _make_tree(tmp_path)
    kept, _ = _validate_globs(["server/sql/*.json", "server/mod/**/*"], root)
    assert kept == ["server/sql/*.json", "server/mod/**/*"]


# --- glob routing: docs-tree globs must not be probed against the code tree -----

def test_partition_routes_docs_glob_to_docs(tmp_path):
    code = str(tmp_path / "code")
    docs = str(tmp_path / "docs_tree")
    code_g, doc_g = _partition_globs(
        ["server/mod/**/*.py",
         f"{docs}/210_design/D031_*.md".replace("\\", "/")],
        code, docs)
    assert code_g == ["server/mod/**/*.py"]
    assert doc_g == [f"{docs}/210_design/D031_*.md".replace("\\", "/")]


def test_abs_under_handles_separators_and_case():
    assert _abs_under("C:/a/b/x.md", "c:\\a\\b")
    assert not _abs_under("C:/a/b/x.md", "C:/a/c")
    assert not _abs_under("rel/x.md", "C:/a")


# --- relative doc globs rooted at docs_root's PARENT (N165) ---------------------

def test_partition_routes_relative_docs_glob_and_strips_basename():
    # Queen emits the design-doc target relative to the workspace root (parent of
    # both trees), carrying docs_root's basename ('Documents') as leading segment.
    # It must route to docs AND be rewritten docs-root-relative so rg matches it.
    code = "C:/workspace/projects/flowgate"
    docs = "C:/workspace/projects/Documents"
    code_g, doc_g = _partition_globs(
        ["client/src/**/*.vue",
         "Documents/projects/FlowGate/210_design/**",
         "Documents/**/FlowGate/**"],
        code, docs)
    assert code_g == ["client/src/**/*.vue"]
    assert doc_g == ["projects/FlowGate/210_design/**", "**/FlowGate/**"]


def test_partition_routes_relative_docs_glob_under_deep_docs_root():
    # T891: the launcher default docs_root is the per-project tree
    # '.../Documents/projects/FlowGate' (basename 'FlowGate'), so the queen's
    # relative glob carries a MULTI-segment overlap ('Documents/projects/FlowGate')
    # whose FIRST segment is 'Documents', not the basename. The single-basename
    # strip never matched it and every doc glob leaked to code (design_excerpts
    # permanently empty). The longest contiguous suffix-overlap must be stripped.
    code = "C:/workspace/projects/FlowGate"
    docs = "C:/workspace/projects/Documents/projects/FlowGate"
    code_g, doc_g = _partition_globs(
        ["client/src/components/**/*.vue",
         "Documents/projects/FlowGate/210_design/**",
         "Documents/projects/FlowGate/**/*.md"],
        code, docs)
    assert code_g == ["client/src/components/**/*.vue"]
    assert doc_g == ["210_design/**", "**/*.md"]


def test_partition_relative_docs_is_noop_without_docs_root():
    # No docs tree → nothing to route to; the relative glob stays code-side (and
    # will be dropped as empty there). Guards against misrouting when --docs absent.
    code_g, doc_g = _partition_globs(
        ["Documents/projects/FlowGate/**"], "C:/code", None)
    assert doc_g == []
    assert code_g == ["Documents/projects/FlowGate/**"]


def test_partition_code_relative_glob_not_mistaken_for_docs():
    # A normal code glob whose first segment differs from docs_root's basename
    # stays code-side.
    code_g, doc_g = _partition_globs(
        ["server/**/*.py", "client/src/**"], "C:/code",
        "C:/workspace/projects/Documents")
    assert doc_g == []
    assert code_g == ["server/**/*.py", "client/src/**"]


def test_retrieve_routes_relative_doc_glob_to_docs_tree(tmp_path):
    # End-to-end of the N165 D030_CHECK/DESIGN_SSOT failure: the doc glob arrives
    # RELATIVE ('Documents/...'), rooted at the parent of the docs tree. It must
    # still pull the design excerpt, not leak to the code tree and vanish.
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True)
    (code / "src" / "view.ts").write_text("mode = 'next'\n", encoding="utf-8")
    docs_parent = tmp_path / "Documents"
    design = docs_parent / "projects" / "FlowGate" / "210_design"
    design.mkdir(parents=True)
    (design / "D030_ssot.md").write_text(
        "# D030\n## section\nin_progress override note\n", encoding="utf-8")
    plan = SearchPlan(
        axis_id="d030_check",
        keywords=["in_progress", "override"],
        file_globs=["Documents/projects/FlowGate/210_design/**"],
        doc_topics=["override"],
    )
    out = retrieve(plan, str(code), str(docs_parent))
    docs_hit = [e["doc"] for e in out["design_excerpts"]]
    assert any("D030_ssot.md" in d for d in docs_hit)


def test_retrieve_routes_doc_glob_to_docs_tree(tmp_path):
    # The queen lowered a design-doc target into file_globs as an ABSOLUTE path
    # under the docs tree. It must be retrieved from docs, not silently dropped
    # for matching nothing under code_root (T890: 0/3 axes located).
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True)
    (code / "src" / "view.ts").write_text("mode = 'next'\n", encoding="utf-8")
    docs = tmp_path / "docs_tree" / "210_design"
    docs.mkdir(parents=True)
    (docs / "D031_ssot.md").write_text(
        "# D031\n## section\nin_progress override note\n", encoding="utf-8")
    plan = SearchPlan(
        axis_id="locate_d031",
        keywords=["in_progress", "override"],
        file_globs=[f"{tmp_path}/docs_tree/210_design/D031_*.md".replace("\\", "/")],
        doc_topics=[],
    )
    out = retrieve(plan, str(code), str(tmp_path / "docs_tree"))
    docs_hit = [e["doc"] for e in out["design_excerpts"]]
    assert any("D031_ssot.md" in d for d in docs_hit)


# --- _widen_globs: drop the wrong-extension constraint, keep directory scope ----

def test_widen_drops_extension_keeping_recursive_dir():
    assert _widen_globs(["client/**/*.js"]) == ["client/**/*"]


def test_widen_basename_extension_becomes_bare_wildcard():
    assert _widen_globs(["*.tsx"]) == ["*"]


def test_widen_specific_file_keeps_name_prefix():
    assert _widen_globs(["a/b/Foo.vue"]) == ["a/b/Foo*"]


def test_widen_extensionless_glob_unchanged():
    assert _widen_globs(["client/**/*"]) == ["client/**/*"]


def test_widen_dedupes_collapsing_globs():
    # *.js and *.tsx both collapse to "*" → a single widened glob.
    assert _widen_globs(["*.js", "*.tsx"]) == ["*"]


# --- end-to-end: the T889 failure mode and its fallback recovery ---------------

def _make_vue_tree(tmp_path):
    """A Vue 3 + TS app: the real site is a .vue component; the only .js is a
    stray build artefact that the keywords never touch (so the queen's .js glob
    matches >0 files — passing the existence probe — yet retrieves nothing)."""
    comp = tmp_path / "client" / "src" / "components"
    comp.mkdir(parents=True)
    (comp / "DocInfoPanel.vue").write_text(
        "<script setup lang='ts'>\n"
        "// isBehindWorkflowHead non-R path force-promotes the badge\n"
        "const effectiveStatus = isBehindWorkflowHead ? 'wf_done' : status\n"
        "</script>\n",
        encoding="utf-8")
    (tmp_path / "client" / "vite.config.js").write_text(
        "export default { plugins: [] }\n", encoding="utf-8")
    return str(tmp_path)


def test_retrieve_recovers_when_queen_globs_wrong_extension(tmp_path):
    # T889: queen emits .js/.tsx globs for a Vue3+TS tree → first pass finds 0
    # snippets → extension-blind fallback widens and recovers the .vue site.
    root = _make_vue_tree(tmp_path)
    plan = SearchPlan(
        axis_id="DIP_BADGE",
        keywords=["isBehindWorkflowHead", "effectiveStatus", "wf_done"],
        file_globs=["client/**/*.js", "*.tsx"],
    )
    out = retrieve(plan, root, max_hops=0)
    files = [s["file"] for s in out["code_snippets"]]
    assert any("DocInfoPanel.vue" in f for f in files)
    widen = out["stats"]["glob_widening"]
    assert widen is not None and widen["recovered"] is True


def test_retrieve_no_widening_when_first_pass_hits(tmp_path):
    # Correct-extension globs find the site directly → no fallback is attempted.
    root = _make_vue_tree(tmp_path)
    plan = SearchPlan(
        axis_id="DIP_BADGE",
        keywords=["isBehindWorkflowHead", "effectiveStatus"],
        file_globs=["client/**/*.vue"],
    )
    out = retrieve(plan, root, max_hops=0)
    assert any("DocInfoPanel.vue" in s["file"] for s in out["code_snippets"])
    assert out["stats"]["glob_widening"] is None


# --- label/i18n discriminator resolution: tell near-identical siblings apart ----
#     (T891) by resolving getLabel('R')->"요건정의" / t('...undecided')->"미정"
#     LOCALLY, so the judge picks by meaning instead of guessing — the cheap read
#     the operator otherwise had to do by hand.

def test_looks_like_label_ref_recognises_getters_and_i18n_keys():
    assert _looks_like_label_ref("docTypeStore.getLabel", "R")
    assert _looks_like_label_ref("t", "workflow.undecided")
    assert _looks_like_label_ref("$t", "a.b.c")
    # a plain function call with a string arg is NOT a label ref (noise to skip).
    assert not _looks_like_label_ref("doStuff", "R")
    assert not _looks_like_label_ref("open", "somefile")


def test_extract_value_reads_label_map_and_locale_shapes():
    assert _extract_value("const labels = { 'R': '요건정의', 'A': '승인' }", "R") == "요건정의"
    assert _extract_value('  "workflow.undecided": "미정",', "workflow.undecided") == "미정"
    assert _extract_value("R => 'requirements'", "R") == "requirements"
    assert _extract_value("unrelated line", "R") is None


def _make_sibling_label_tree(tmp_path):
    """Two near-identical sibling elements distinguishable ONLY by their label
    reference — the T891 shape — plus the local store/locale that resolve them."""
    comp = tmp_path / "client" / "src" / "components"
    comp.mkdir(parents=True)
    (comp / "DocWorkflow.vue").write_text(
        "<template>\n"
        "  <div class='wf-undecided'>\n"
        "    <span>{{ docTypeStore.getLabel('R') }}</span>\n"
        "  </div>\n"
        "  <div class='wf-undecided'>\n"
        "    <span>{{ t('workflow.undecided') }}</span>\n"
        "  </div>\n"
        "</template>\n",
        encoding="utf-8")
    store = tmp_path / "client" / "src" / "stores"
    store.mkdir(parents=True)
    (store / "docType.ts").write_text(
        "const labels = { 'R': '요건정의', 'A': '승인' }\n", encoding="utf-8")
    locale = tmp_path / "client" / "src" / "locales"
    locale.mkdir(parents=True)
    (locale / "ko.json").write_text(
        '{ "workflow.undecided": "미정" }\n', encoding="utf-8")
    return str(tmp_path)


def test_retrieve_resolves_sibling_label_discriminators(tmp_path):
    root = _make_sibling_label_tree(tmp_path)
    plan = SearchPlan(
        axis_id="WF_HL",
        keywords=["wf-undecided", "getLabel", "undecided"],
        file_globs=["client/**/*.vue"],
    )
    out = retrieve(plan, root, max_hops=0)
    resolved: dict[str, list[str]] = {}
    for s in out["code_snippets"]:
        for r in s.get("resolved", []):
            resolved[r["key"]] = r["values"]
    # both sibling labels resolved to their rendered text — the bundle now CARRIES
    # the discriminator, so a downstream judge need not guess (or read source).
    assert resolved.get("R") == ["요건정의"]
    assert resolved.get("workflow.undecided") == ["미정"]
    assert out["stats"]["resolved_refs"] >= 2


def test_retrieve_does_not_invent_values_for_unresolvable_refs(tmp_path):
    # No store/locale defines the key → nothing is attached (we never hallucinate
    # a value; absence keeps the bundle honest).
    comp = tmp_path / "src"
    comp.mkdir(parents=True)
    (comp / "X.vue").write_text(
        "<span>{{ getLabel('ZZZ_UNDEFINED') }}</span>\n", encoding="utf-8")
    plan = SearchPlan(axis_id="X", keywords=["getLabel"], file_globs=["src/**/*.vue"])
    out = retrieve(plan, str(tmp_path), max_hops=0)
    assert out["stats"]["resolved_refs"] == 0
    assert all("resolved" not in s for s in out["code_snippets"])


# ── Defect 1 (T892): dense one-record-per-line files (SQL/JSON query maps) ──────
def _make_dense_sql_tree(tmp_path):
    qdir = tmp_path / "server" / "sql" / "queries"
    qdir.mkdir(parents=True)
    # one SQL query per line, each a long single-line string — the queries.json
    # shape where get_pending_head_by_group sat on a 586-char line (T892).
    lines = [
        "{",
        '  "get_in_progress_head_by_group": "SELECT wsi.*, ws.id FROM workflow_sequence_items'
        " wsi JOIN workflow_sequences ws ON wsi.sequence_id = ws.id WHERE d.group_id = ?"
        ' AND wsi.result_doc_id IS NOT NULL ORDER BY wsi.sort_order ASC LIMIT 1",',
        '  "get_pending_head_by_group": "SELECT wsi.*, ws.id FROM workflow_sequence_items'
        " wsi JOIN workflow_sequences ws ON wsi.sequence_id = ws.id WHERE d.group_id = ?"
        " AND d.project_id = ? AND wsi.result_doc_id IS NULL ORDER BY wsi.sort_order ASC"
        ' LIMIT 1",',
        '  "get_effective_head": "SELECT wsi.* FROM workflow_sequence_items wsi WHERE'
        ' d.group_id = ? ORDER BY wsi.sort_order ASC LIMIT 1"',
        "}",
    ]
    (qdir / "queries.json").write_text("\n".join(lines), encoding="utf-8")
    return str(tmp_path)


def test_retrieve_surfaces_exact_line_in_dense_single_line_file(tmp_path):
    # The ±k window + cluster picking could not isolate the target key on a dense
    # one-query-per-line file (every line carries SELECT/WHERE/JOIN, so dozens tie),
    # so the judge ruled located=False on the SAME file another run located. The
    # dense-line path must surface get_pending_head_by_group as its OWN snippet.
    root = _make_dense_sql_tree(tmp_path)
    plan = SearchPlan(
        axis_id="H",
        keywords=["get_pending_head_by_group", "result_doc_id", "ORDER BY", "IS NULL"],
        file_globs=["server/sql/queries/queries.json"],
    )
    out = retrieve(plan, root, max_hops=0)
    snips = out["code_snippets"]
    target = [s for s in snips if "get_pending_head_by_group" in s["text"]]
    assert target, f"target line not surfaced: {[(s['lines'], s['text'][:40]) for s in snips]}"
    s = target[0]
    lo, hi = s["lines"].split("-")
    assert lo == hi, f"expected a single-record snippet, got {s['lines']}"
    # the WHOLE SQL string is present (not truncated out of a ±k window)
    assert "result_doc_id IS NULL" in s["text"]
    assert "ORDER BY wsi.sort_order" in s["text"]


def test_dense_file_does_not_collapse_distinct_query_lines(tmp_path):
    # Each distinct query line is its own snippet (not merged into one whole-file
    # window) so the judge can tell get_pending from get_in_progress.
    root = _make_dense_sql_tree(tmp_path)
    plan = SearchPlan(
        axis_id="H",
        keywords=["result_doc_id", "ORDER BY", "LIMIT"],
        file_globs=["server/sql/queries/queries.json"],
    )
    out = retrieve(plan, root, max_hops=0)
    # at least the two result_doc_id-bearing queries surface as separate records
    rec_lines = {s["lines"] for s in out["code_snippets"]}
    assert len(rec_lines) >= 2, rec_lines


# ── HTTP call-binding edge (N177): client fetch-URL literal → backend route handler.
#    The crux miss — the judge grounded on a lexically-similar but WRONG handler
#    (get_effective_head, literally "head") because the only path to the real source
#    (_parse_doc_workflow, reached via /api/v1/documents/detail) is a cross-language
#    URL-literal→route string join that keyword retrieval never builds.

def test_path_segs_strips_and_splits():
    assert _path_segs("/api/v1/documents/detail") == ["api", "v1", "documents", "detail"]
    assert _path_segs("/detail/") == ["detail"]
    assert _path_segs("/") == []


def test_route_suffix_match_literal_tail():
    # decorator carries only the router-relative tail; the mount prefix lives elsewhere.
    url = _path_segs("/api/v1/documents/detail")
    assert _route_suffix_match(url, _path_segs("/detail")) == (True, 1, 0)


def test_route_suffix_match_param_segment_matches_any():
    url = _path_segs("/api/v1/workflow/D031/head")
    # {doc_id} param matches the "D031" segment; "head" matches literally.
    assert _route_suffix_match(url, _path_segs("/workflow/{doc_id}/head")) == (True, 2, 1)


def test_route_suffix_match_rejects_non_suffix():
    url = _path_segs("/api/v1/documents/detail")
    assert _route_suffix_match(url, _path_segs("/workflow/{doc_id}/head"))[0] is False
    # longer than url → no match
    assert _route_suffix_match(["detail"], ["a", "detail"])[0] is False


def _make_fe_be_binding_tree(tmp_path):
    """N177 in miniature: a FE bar reads workflow_head_type from /documents/detail;
    a decoy BE route /workflow/{doc_id}/head returns get_effective_head (includes M);
    the real source /detail → _parse_doc_workflow (excludes M) is reachable only via
    the URL literal. There is also a /{doc_id} param route the param-match must lose to.
    """
    fe = tmp_path / "client" / "src" / "components"
    fe.mkdir(parents=True)
    (fe / "DocHeader.vue").write_text(
        "async function fetchDoc(id) {\n"
        "  const res = await getRequest(`/api/v1/documents/detail?doc_id=${id}`)\n"
        "  doc.value = res.data\n"
        "}\n"
        "const workflowHeadType = computed(() => doc.value?.workflow_head_type ?? null)\n",
        encoding="utf-8")
    be = tmp_path / "server" / "routers"
    be.mkdir(parents=True)
    # decoy: the lexically-obvious "head" route — what keyword retrieval grabs.
    (be / "workflow_head_routes.py").write_text(
        'router = APIRouter(prefix="/api/v1")\n'
        '@router.get("/workflow/{doc_id}/head")\n'
        "def get_workflow_head(doc_id):\n"
        "    head = get_effective_head(seq_id)  # includes M\n"
        "    return {'workflow_head_type': head.get('type')}\n",
        encoding="utf-8")
    # real source: reached only by matching the /detail URL literal.
    (be / "documents.py").write_text(
        'router = APIRouter(prefix="/api/v1/documents")\n'
        '@router.get("/{doc_id}")\n'
        "def get_document(doc_id):\n"
        "    return _parse_doc_workflow(doc)\n"
        "\n"
        '@router.get("/detail")\n'
        "@require_permission('perm_document_read')\n"
        "def get_document_rpc(doc_id):\n"
        "    return _parse_doc_workflow(doc)  # NON_HEAD_TYPES excludes M\n",
        encoding="utf-8")
    return str(tmp_path)


def test_resolve_http_bindings_lands_on_real_source_not_decoy(tmp_path):
    root = _make_fe_be_binding_tree(tmp_path)
    snippets = [{
        "file": "client/src/components/DocHeader.vue",
        "lines": "1-5",
        "text": "const res = await getRequest(`/api/v1/documents/detail?doc_id=${id}`)\n",
    }]
    bindings = _resolve_http_bindings(snippets, root)
    assert bindings, "no binding resolved for the /detail fetch"
    b = bindings[0]
    # the literal /detail route wins over the /{doc_id} param route (FastAPI precedence)
    assert b["route"] == "/detail"
    assert "documents.py" in b["file"]
    # the handler body the judge now sees points at the REAL source, not the decoy.
    assert "_parse_doc_workflow" in b["text"]
    assert "get_effective_head" not in b["text"]
    assert b["via"] == "http-binding"
    assert b["ambiguous"] is False


def test_retrieve_attaches_http_binding_to_call_chain(tmp_path):
    # End-to-end: the wired retrieve() surfaces the resolved handler in call_chain
    # so the judge consumes it with the rest of the evidence (no judge.py change).
    root = _make_fe_be_binding_tree(tmp_path)
    plan = SearchPlan(
        axis_id="WF_HEAD",
        keywords=["workflow_head_type", "getRequest", "fetchDoc"],
        file_globs=["client/**/*.vue"],
    )
    out = retrieve(plan, root, max_hops=0)
    assert out["stats"]["http_bindings"] >= 1, out["stats"]
    handlers = [b for b in out["http_bindings"] if b["route"] == "/detail"]
    assert handlers and "_parse_doc_workflow" in handlers[0]["text"]
    # the resolved handler rides along in call_chain (what summarize_bundle renders).
    assert any(s.get("via") == "http-binding" and "_parse_doc_workflow" in s.get("text", "")
               for s in out["call_chain"])


def test_resolve_http_bindings_ignores_non_http_callees(tmp_path):
    # A non-fetch callee taking a "/x" string literal must NOT be treated as a
    # binding (we only resolve known HTTP request callees — same tight discipline
    # as label getters), else the bundle floods with noise.
    root = _make_fe_be_binding_tree(tmp_path)
    snippets = [{
        "file": "x.ts", "lines": "1-1",
        "text": "const p = path.join('/api/v1/documents/detail')\n"
                "logger.info('/api/v1/documents/detail')\n",
    }]
    assert _resolve_http_bindings(snippets, root) == []


def test_resolve_http_bindings_surfaces_duplicate_sources_as_ambiguous(tmp_path):
    # Two distinct handlers declared for the SAME client path → ambiguous; BOTH are
    # surfaced (never guess) — and duplicated sources for one screen value is itself
    # the tangle worth flagging (the user's "millionth fix" churn).
    fe = tmp_path / "client"
    fe.mkdir(parents=True)
    (fe / "api.ts").write_text(
        "export const load = () => axios.get('/api/items/list')\n", encoding="utf-8")
    be = tmp_path / "server"
    be.mkdir(parents=True)
    (be / "a.py").write_text(
        '@router.get("/list")\ndef list_a():\n    return source_a()\n', encoding="utf-8")
    (be / "b.py").write_text(
        '@router.get("/list")\ndef list_b():\n    return source_b()\n', encoding="utf-8")
    snippets = [{"file": "client/api.ts", "lines": "1-1",
                 "text": "axios.get('/api/items/list')\n"}]
    bindings = _resolve_http_bindings(snippets, str(tmp_path))
    routes = {b["route"] for b in bindings}
    assert routes == {"/list"}
    assert len(bindings) == 2, [b["file"] for b in bindings]
    assert all(b["ambiguous"] for b in bindings)


def test_resolve_http_bindings_prefix_disambiguates_same_tail(tmp_path):
    # Two routers declare the SAME decorator tail (/detail) but different prefixes;
    # only the one whose prefix completes the client URL must resolve (FlowGate's
    # real /documents vs /api/v1 legacy collision). Reading each router's OWN
    # declared prefix breaks the tie deterministically — not a false ambiguity.
    fe = tmp_path / "client"
    fe.mkdir(parents=True)
    (fe / "api.ts").write_text(
        "const r = await getRequest('/api/v1/documents/detail?id=1')\n", encoding="utf-8")
    be = tmp_path / "server"
    be.mkdir(parents=True)
    (be / "documents.py").write_text(
        'router = APIRouter(prefix="/documents", tags=["Documents"])\n'
        '@router.get("/detail")\ndef doc_detail():\n    return real_source()\n',
        encoding="utf-8")
    (be / "legacy.py").write_text(
        'router = APIRouter(prefix="/api/v1", tags=["Legacy"])\n'
        '@router.get("/detail")\ndef legacy_detail():\n    return legacy_source()\n',
        encoding="utf-8")
    snippets = [{"file": "client/api.ts", "lines": "1-1",
                 "text": "getRequest('/api/v1/documents/detail?id=1')\n"}]
    bindings = _resolve_http_bindings(snippets, str(tmp_path))
    assert len(bindings) == 1, [(b["file"], b["full_path"]) for b in bindings]
    b = bindings[0]
    assert "documents.py" in b["file"] and b["full_path"] == "/documents/detail"
    assert b["ambiguous"] is False
    assert "real_source" in b["text"] and "legacy_source" not in b["text"]


def test_resolve_http_bindings_folds_outer_mount_prefix_into_route(tmp_path):
    # M036 ground truth. The live route gets /api/v1 only where main.py mounts its
    # prefix-less router; a legacy decoy declares /api/v1 itself. Both therefore
    # represent the same full path and must tie rather than dropping the live
    # producer. Crucially this mirrors FlowGate's REAL layout: code_root is the
    # repo root but the python source root is the ``server/`` SUBDIR, so the mount
    # wiring imports ``from modules.flow_gate...`` (no ``server.`` prefix). The
    # mount-prefix fold must resolve that absolute import by unique path suffix —
    # an exact-root match would miss it and silently fall back to the buggy
    # suffix-only score that ranks the dead decoy above the live handler.
    fe = tmp_path / "client"
    fe.mkdir(parents=True)
    (fe / "api.ts").write_text(
        "const projects = await getRequest('/api/v1/projects')\n",
        encoding="utf-8")

    settings = (tmp_path / "server" / "modules" / "flow_gate" / "settings"
                / "routers")
    settings.mkdir(parents=True)
    (settings / "project_settings.py").write_text(
        'router = APIRouter(tags=["Project Settings"])\n'
        '@router.get("/projects")\n'
        "def live_projects():\n"
        "    return live_source()\n",
        encoding="utf-8")

    legacy = tmp_path / "server" / "modules" / "flow_gate" / "api" / "v1"
    legacy.mkdir(parents=True)
    (legacy / "legacy_misc_routes.py").write_text(
        'router = APIRouter(prefix="/api/v1")\n'
        '@router.get("/projects")\n'
        "def legacy_projects():\n"
        "    return legacy_source()\n",
        encoding="utf-8")

    wiring = tmp_path / "server" / "routers"
    wiring.mkdir(parents=True)
    (wiring / "main.py").write_text(
        "from modules.flow_gate.settings.routers.project_settings "
        "import router as _settings_project_router\n"
        'app.include_router(_settings_project_router, '
        'prefix=f"{CONTEXT}/api/v1")\n',
        encoding="utf-8")

    snippets = [{"file": "client/api.ts", "lines": "1-1",
                 "text": "getRequest('/api/v1/projects')\n"}]
    bindings = _resolve_http_bindings(snippets, str(tmp_path))

    assert len(bindings) == 2, [
        (b["file"], b["full_path"]) for b in bindings
    ]
    assert {b["full_path"] for b in bindings} == {"/api/v1/projects"}
    assert {os.path.basename(b["file"]) for b in bindings} == {
        "project_settings.py", "legacy_misc_routes.py",
    }
    assert all(b["ambiguous"] for b in bindings)


def test_http_binding_uses_first_mounted_duplicate_and_traces_its_producer(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "__init__.py").write_text("", encoding="utf-8")
    (app / "first.py").write_text(
        'from fastapi import APIRouter\n'
        'from app.data import list_projects\n'
        'router = APIRouter(prefix="/api/v1")\n'
        '@router.get("/projects")\n'
        'def first_projects():\n'
        '    projects = list_projects()\n'
        '    return {"projects": projects}\n',
        encoding="utf-8")
    (app / "second.py").write_text(
        'from fastapi import APIRouter\n'
        'router = APIRouter(prefix="/api/v1")\n'
        '@router.get("/projects")\n'
        'def second_projects():\n'
        '    return {"projects": off_path_projects()}\n'
        'def off_path_projects():\n'
        '    return []\n',
        encoding="utf-8")
    (app / "data.py").write_text(
        'def list_projects():\n'
        '    return store._fetch_all("SELECT * FROM projects ORDER BY project_id")\n',
        encoding="utf-8")
    (app / "main.py").write_text(
        'from fastapi import FastAPI\n'
        'from app.first import router as first_router\n'
        'from app.second import router as second_router\n'
        'app = FastAPI()\n'
        'app.include_router(first_router)\n'
        'app.include_router(second_router)\n',
        encoding="utf-8")
    snippets = [{"file": "client/view.ts", "lines": "1",
                 "text": "client.get('/api/v1/projects')"}]

    bindings = _resolve_http_bindings(snippets, str(tmp_path))
    assert len(bindings) == 1
    assert bindings[0]["file"].endswith("app/first.py")
    assert bindings[0]["winning"] is True
    assert bindings[0]["ambiguous"] is False
    assert bindings[0]["shadowed"][0]["file"].endswith("app/second.py")

    producers = _resolve_http_producer_paths(bindings, str(tmp_path))
    assert any(p["file"].endswith("app/data.py") and p["producer"]
               for p in producers)
    assert all("off_path_projects" not in p.get("text", "") for p in producers)


def test_read_def_body_reads_past_multiline_signature(tmp_path):
    # Regression: a multi-line def signature whose closing ")" sits at the def's own
    # indent stopped the body read INSIDE the signature, so the real work below was
    # never read (FlowGate get_document → _parse_doc_workflow at +10 lines was lost).
    f = tmp_path / "m.py"
    f.write_text(
        "def get_document(\n"
        "    doc_id: str,\n"
        "    current_user: dict = Depends(get_current_user),\n"
        ") -> dict:\n"
        '    """Fetch a single document."""\n'
        "    doc = service.get(doc_id)\n"
        "    return _parse_doc_workflow(doc)\n"
        "\n"
        "def other():\n"
        "    pass\n",
        encoding="utf-8")
    out = _read_def_body(str(tmp_path), "m.py", 1)
    assert "_parse_doc_workflow(doc)" in out["text"], out["text"]
    assert "def other" not in out["text"]  # stops at the sibling def


# ── PEER-IMPLEMENTATION grounding (sibling-pattern resolver).
#    The live FlowGate N175 z-index shape: a toast overlay with z-index:2000 sits
#    behind a modal because it does NOT escape its stacking context, while a sibling
#    overlay in the SAME common/ folder uses <Teleport to="body">. The fix is in the
#    asymmetry between the two files, not in either file's number — invisible to
#    keyword/density retrieval.

def _make_stacking_tree(tmp_path):
    comp = tmp_path / "client" / "src" / "components" / "common"
    comp.mkdir(parents=True)
    # target: overlay with a HIGH z-index but no teleport (the buggy toast).
    (comp / "ToastContainer.vue").write_text(
        "<template>\n  <div class=\"toast-host\"><slot/></div>\n</template>\n"
        "<style scoped>\n.toast-host { position: fixed; z-index: 2000; }\n</style>\n",
        encoding="utf-8")
    # sibling overlay that DOES escape via Teleport — the corrective pattern.
    (comp / "ContextMenu.vue").write_text(
        "<template>\n  <Teleport to=\"body\">\n    <ul class=\"menu\"/>\n  </Teleport>\n</template>\n"
        "<style scoped>\n.menu { position: absolute; z-index: 1500; }\n</style>\n",
        encoding="utf-8")
    # a non-overlay sibling (no position/z-index) must be ignored as incomparable.
    (comp / "PlainButton.vue").write_text(
        "<template>\n  <button><slot/></button>\n</template>\n", encoding="utf-8")
    return str(tmp_path)


def test_peer_pattern_surfaces_teleport_asymmetry(tmp_path):
    root = _make_stacking_tree(tmp_path)
    snippets = [{
        "file": "client/src/components/common/ToastContainer.vue",
        "lines": "4-5",
        "text": ".toast-host { position: fixed; z-index: 2000; }\n",
    }]
    out = _resolve_peer_patterns(snippets, root)
    assert out, "no peer pattern surfaced for the toast overlay"
    p = out[0]
    assert p["via"] == "peer-pattern" and p["concern"] == "stacking"
    # the corrective sibling (teleport) is named; the incomparable plain one is not.
    assert any("ContextMenu.vue" in s for s in p["siblings"])
    assert all("PlainButton.vue" not in s for s in p["siblings"])
    # the rendered block tells the judge the real lever is Teleport, not the number.
    assert "Teleport" in p["text"] and "teleport=NO" in p["text"]
    assert "teleport=YES" in p["text"]


def test_peer_pattern_silent_when_siblings_agree(tmp_path):
    # If every comparable overlay escapes the same way as the target, there is no
    # asymmetry to show → resolver stays silent (gate 2 = "only when needed").
    comp = tmp_path / "client" / "src" / "components"
    comp.mkdir(parents=True)
    (comp / "ToastContainer.vue").write_text(
        "<template><Teleport to=\"body\"><div/></Teleport></template>\n"
        "<style>.t { position: fixed; z-index: 2000; }</style>\n", encoding="utf-8")
    (comp / "ContextMenu.vue").write_text(
        "<template><Teleport to=\"body\"><ul/></Teleport></template>\n"
        "<style>.m { position: absolute; z-index: 1500; }</style>\n", encoding="utf-8")
    snippets = [{"file": "client/src/components/ToastContainer.vue",
                 "lines": "2-2", "text": "z-index: 2000;"}]
    assert _resolve_peer_patterns(snippets, str(tmp_path)) == []


def test_peer_pattern_silent_for_non_overlay_target(tmp_path):
    # Gate 1: a file with no stacking signal at all never triggers the resolver,
    # even if a sibling is a teleporting overlay.
    comp = tmp_path / "src"
    comp.mkdir(parents=True)
    (comp / "Plain.vue").write_text(
        "<template><button/></template>\n", encoding="utf-8")
    (comp / "Menu.vue").write_text(
        "<template><Teleport to=\"body\"><ul/></Teleport></template>\n"
        "<style>.m { position: absolute; z-index: 9; }</style>\n", encoding="utf-8")
    snippets = [{"file": "src/Plain.vue", "lines": "1-1", "text": "<button/>"}]
    assert _resolve_peer_patterns(snippets, str(tmp_path)) == []


def test_peer_pattern_ignores_non_component_files(tmp_path):
    # A backend .py handler riding in call_chain (e.g. an http-binding result) must
    # not be probed for stacking peers — the registry is component-class only.
    be = tmp_path / "server"
    be.mkdir(parents=True)
    (be / "h.py").write_text("def f():\n    return {'z-index': 2000}\n", encoding="utf-8")
    snippets = [{"file": "server/h.py", "lines": "1-2", "text": "z-index 2000"}]
    assert _resolve_peer_patterns(snippets, str(tmp_path)) == []


def test_stacking_profile_reads_css_and_js_zindex():
    css = _stacking_profile(".x { position: fixed; z-index: 2000; }")
    assert css["applies"] and css["zindex"] == 2000 and css["position"] == "fixed"
    assert css["teleports"] is False
    js = _stacking_profile("const s = { zIndex: 50 }; createPortal(x, document.body)")
    assert js["applies"] and js["zindex"] == 50 and js["teleports"] is True


def test_peer_pattern_attaches_to_call_chain(tmp_path):
    # End-to-end: wired retrieve() surfaces the asymmetry in call_chain so the judge
    # consumes it with the rest of the evidence (no judge.py change).
    root = _make_stacking_tree(tmp_path)
    plan = SearchPlan(
        axis_id="TOAST_ZINDEX",
        keywords=["toast-host", "z-index", "position"],
        file_globs=["client/**/*.vue"],
    )
    out = retrieve(plan, root, max_hops=0)
    assert out["stats"]["peer_patterns"] >= 1, out["stats"]
    assert any(s.get("via") == "peer-pattern" and "Teleport" in s.get("text", "")
               for s in out["call_chain"])


# ── FETCH-HARVEST grounding: surface the fetch that feeds an "empty UI variable"
# symptom even when keyword density landed BETWEEN the windows and missed it, so the
# producer chain (route → service → store query) reaches the converger. Mirrors the
# live FlowGate "module selector never renders" miss: currentModules' keyword cluster
# sat on the assignment/render while the getRequest that feeds it fell in a gap, so the
# /api/v1/projects → get_projects_with_modules → store ("'' AS module") chain never
# entered the bundle and the converger stranded on symptom-side red herrings.

def _make_empty_var_tree(tmp_path):
    """FE assigns an 'empty' var from a fetch placed in a KEYWORD GAP; BE producer
    chain bottoms out at a store query that hardcodes the field empty."""
    fe = tmp_path / "client" / "src" / "components"
    fe.mkdir(parents=True)
    gap = "\n".join(f"  // step {i}: massage the response payload" for i in range(12))
    (fe / "NewReqModal.vue").write_text(
        "const currentModules = ref([])\n"                       # L1: keyword hit
        "\n"
        "async function load() {\n"
        + gap + "\n"                                             # padding (no keyword)
        "  const res = await getRequest('/api/v1/projects')\n"   # fetch in the GAP
        + gap + "\n"                                             # padding (no keyword)
        "  const selected = res.data.projects[0]\n"
        "  currentModules.value = selected.modules\n"            # symptom assignment
        "}\n",
        encoding="utf-8")
    be = tmp_path / "server" / "routes"
    be.mkdir(parents=True)
    (be / "list_routes.py").write_text(
        'router = APIRouter(prefix="/api/v1")\n'
        '@router.get("/projects")\n'
        "def api_projects():\n"
        "    return {'projects': get_projects_with_modules()}\n",
        encoding="utf-8")
    svc = tmp_path / "server" / "svc"
    svc.mkdir(parents=True)
    (svc / "process_service.py").write_text(
        "def get_projects_with_modules():\n"
        "    rows = get_allowed_projects()\n"
        "    return rows\n",
        encoding="utf-8")
    store = tmp_path / "server" / "db"
    store.mkdir(parents=True)
    (store / "store.py").write_text(
        "def get_allowed_projects():\n"
        "    # BUG: module column is hardcoded empty — every project gets no modules\n"
        "    return run(\"SELECT project_id, project_name, '' AS module FROM projects\")\n",
        encoding="utf-8")
    return str(tmp_path)


def test_covered_ranges_parses_line_spans():
    snips = [{"file": "a.vue", "lines": "10-20"}, {"file": "a.vue", "lines": "30"},
             {"file": "b.vue", "lines": "5-7"}]
    assert _covered_ranges(snips, "a.vue") == [(10, 20), (30, 30)]
    assert _covered_ranges(snips, "b.vue") == [(5, 7)]


def test_harvest_surfaces_fetch_in_keyword_gap(tmp_path):
    root = _make_empty_var_tree(tmp_path)
    # The keyword windows cover the currentModules mentions (top and bottom) but NOT
    # the getRequest in the middle gap — exactly the live miss.
    snippets = [
        {"file": "client/src/components/NewReqModal.vue", "lines": "1-3"},
        {"file": "client/src/components/NewReqModal.vue", "lines": "28-30"},
    ]
    # Without harvest the fetch URL is invisible → no binding can cross to the backend.
    assert _resolve_http_bindings(snippets, root) == []
    harvest = _harvest_inscope_fetch_urls(snippets, root)
    assert any("/api/v1/projects" in h["text"] for h in harvest), harvest
    assert all(h["via"] == "fetch-harvest" for h in harvest)
    # With the harvested fetch in scope, the boundary resolver crosses to the handler.
    bindings = _resolve_http_bindings(snippets + harvest, root)
    assert any(b["route"] == "/projects" for b in bindings), bindings


def test_harvest_skips_fetch_already_in_scope(tmp_path):
    root = _make_empty_var_tree(tmp_path)
    # When a window already covers the fetch line, harvest must NOT duplicate it.
    full = [{"file": "client/src/components/NewReqModal.vue", "lines": "1-30"}]
    assert _harvest_inscope_fetch_urls(full, root) == []


def test_retrieve_reaches_store_query_through_harvested_fetch(tmp_path):
    # End-to-end: a module-keyword axis that misses the fetch still unrolls the full
    # producer chain to the store query that hardcodes the field empty (the real defect).
    root = _make_empty_var_tree(tmp_path)
    plan = SearchPlan(
        axis_id="EMPTY_MODULES",
        keywords=["currentModules", "modules"],
        file_globs=["client/**/*.vue"],
    )
    out = retrieve(plan, root)
    assert out["stats"]["fetch_harvest"] >= 1, out["stats"]
    chain_text = "\n".join(s.get("text", "") for s in out["call_chain"])
    assert "'' AS module" in chain_text, "producer chain did not reach the store query"
    assert "get_projects_with_modules" in chain_text


# ── FIELD-PRODUCER grounding (N183 round-2) ─────────────────────────────────────

def _make_head_field_tree(tmp_path):
    """An FE that reads the snake_case response field ``workflow_head_type`` and a
    backend that FILLS it next to the buggy ``NON_HEAD_TYPES`` exclusion — plus a SQL
    helper merely NAMED ``get_effective_head`` (the lexical decoy, never serializes the
    field). Mirrors the live head-strip off-by-one (M035 §4)."""
    fe = tmp_path / "client" / "src" / "components"
    fe.mkdir(parents=True)
    (fe / "DocHeader.vue").write_text(
        "<script setup>\n"
        "// the FE reads the field exactly as the server serialized it (snake_case)\n"
        "const workflowHeadType = computed(() => doc.value?.workflow_head_type ?? null)\n"
        "</script>\n",
        encoding="utf-8")
    be = tmp_path / "server" / "documents" / "routers"
    be.mkdir(parents=True)
    (be / "documents.py").write_text(
        "def build_doc_detail(doc, out, seq_items):\n"
        "    if seq_items:\n"
        '        NON_HEAD_TYPES = {"R", "M", "Q"}\n'
        "        head_type = next(\n"
        '            (it["type"] for it in seq_items\n'
        '             if it["type"] not in NON_HEAD_TYPES),\n'
        "            None,\n"
        "        )\n"
        "        if head_type is not None:\n"
        '            out["workflow_head_type"] = head_type\n'
        "    return out\n",
        encoding="utf-8")
    sql = tmp_path / "server" / "sql"
    sql.mkdir(parents=True)
    (sql / "queries.py").write_text(
        "def get_effective_head(project_id):\n"
        "    # lexical decoy: named 'head' but the FE never reads this; fills nothing\n"
        '    return run("SELECT type FROM seq ORDER BY result_doc_id")\n',
        encoding="utf-8")
    return str(tmp_path)


def test_field_producer_surfaces_filler_not_named_decoy(tmp_path):
    root = _make_head_field_tree(tmp_path)
    # only the FE read is in scope (the symptom side); the producer is off-path.
    snippets = [{"file": "client/src/components/DocHeader.vue", "lines": "1-4",
                 "text": "const workflowHeadType = doc.value?.workflow_head_type ?? null"}]
    out = _resolve_field_producers(snippets, root)
    assert out, "field-producer grounding surfaced nothing"
    assert all(p["via"] == "field-producer" for p in out)
    text = "\n".join(p["text"] for p in out)
    # the real filler AND the buggy exclusion it sits next to are now in scope …
    assert 'out["workflow_head_type"] = head_type' in text
    assert "NON_HEAD_TYPES" in text
    # … and the lexical decoy (named 'head', serializes nothing) is NOT surfaced.
    assert "get_effective_head" not in text
    assert all("queries.py" not in p["file"] for p in out)


def test_field_producer_ignores_reads_and_type_decls(tmp_path):
    # A field that only ever appears as a READ or a type-decl (never serialized) has no
    # producing site → nothing to surface (the quote+[:=] shape is the gate).
    d = tmp_path / "client"
    d.mkdir()
    (d / "Doc.vue").write_text(
        "const x = doc.value?.workflow_head_type\n"
        "interface H { workflow_head_type?: string | null }\n"
        'const y = resp.get("workflow_head_type")\n',
        encoding="utf-8")
    snippets = [{"file": "client/Doc.vue", "lines": "1-3",
                 "text": "doc.value?.workflow_head_type interface workflow_head_type?: string"}]
    assert _resolve_field_producers(snippets, str(tmp_path)) == []


def test_field_producer_skips_camelcase_and_short_tokens(tmp_path):
    (tmp_path / "a.vue").write_text('out["headType"] = x\nout["id"] = y\n', encoding="utf-8")
    # camelCase ``headType`` and the short bare word ``id`` are never snake_case fields.
    snippets = [{"file": "client/a.vue", "lines": "1-2", "text": "headType id docId"}]
    assert _resolve_field_producers(snippets, str(tmp_path)) == []


def test_field_producer_harvests_only_from_fe_files(tmp_path):
    # The producer exists; the only difference is which file the field is READ from.
    (tmp_path / "svc.py").write_text('out["sort_order"] = compute()\n', encoding="utf-8")
    # snake_case in a SERVER file is a DB column / local, not a FE-bound field → ignored.
    server_snip = [{"file": "server/svc.py", "lines": "1", "text": "sort_order = compute()"}]
    assert _resolve_field_producers(server_snip, str(tmp_path)) == []
    # the SAME token read from a FE file is a response field → resolves to the producer.
    fe_snip = [{"file": "client/x.vue", "lines": "1", "text": "const o = row.sort_order"}]
    out = _resolve_field_producers(fe_snip, str(tmp_path))
    assert any(p["field"] == "sort_order" for p in out), out


def test_field_producer_skips_producer_already_in_scope(tmp_path):
    root = _make_head_field_tree(tmp_path)
    # When a window already covers the producer file/line, do not duplicate it.
    snippets = [
        {"file": "client/src/components/DocHeader.vue", "lines": "1-4",
         "text": "workflow_head_type"},
        {"file": "server/documents/routers/documents.py", "lines": "1-12",
         "text": 'out["workflow_head_type"] = head_type'},
    ]
    out = _resolve_field_producers(snippets, root)
    assert all("documents.py" not in p["file"] for p in out), out


def test_field_producer_drops_overcommon_key(tmp_path):
    # A field serialized in too many places is a common key, not a pinpoint → dropped.
    lines = "".join(f'd{i}["project_id_field"] = {i}\n' for i in range(8))
    (tmp_path / "many.py").write_text(lines, encoding="utf-8")
    snippets = [{"file": "x.vue", "lines": "1", "text": "project_id_field"}]
    assert _resolve_field_producers(snippets, str(tmp_path)) == []
