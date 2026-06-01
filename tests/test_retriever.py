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
