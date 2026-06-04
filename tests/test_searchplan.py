"""Unit tests for hive.searchplan — the queen→retrieve bridge.

Pure text extraction, no model calls. Pins the deterministic (B) extractor and
the hybrid (A∪B) merge so the seed quality the judge depends on is regression-safe.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive.searchplan import (
    coverage_risk, extract_doc_topics, extract_globs, extract_keywords,
    task_to_searchplan, is_visibility_symptom, with_visibility_probe,
)
from hive.retriever import SearchPlan


class TestGlobExtraction(unittest.TestCase):
    def test_keeps_real_repo_paths(self):
        brief = ("Search server/sql/queries/*.json and "
                 "server/modules/flow_gate/db/ for the head query.")
        globs = extract_globs(brief)
        self.assertIn("server/sql/queries/*.json", globs)
        # bare directory becomes a recursive subtree scope
        self.assertIn("server/modules/flow_gate/db/**/*", globs)

    def test_exact_file_path_passes_through(self):
        self.assertEqual(extract_globs("see server/routers/main.py here"),
                         ["server/routers/main.py"])

    def test_drops_prose_slashes(self):
        # "head/lookup", "and/or", "pros/cons" are not paths
        self.assertEqual(extract_globs("the head/lookup pros/cons and/or split"), [])

    def test_depth_two_non_root_still_path(self):
        self.assertEqual(extract_globs("look in foo/bar/baz module"),
                         ["foo/bar/baz/**/*"])

    def test_doc_code_run_is_not_a_path(self):
        # slash-joined doc codes ("D0xx/R0xx/TR863") are not a directory path
        self.assertEqual(extract_globs("requirements D0xx/R0xx/M0xx/TR866/TR863"), [])


class TestKeywordExtraction(unittest.TestCase):
    def test_code_identifiers(self):
        kw = extract_keywords("trace workflowViewState into ReviewActionBar; "
                              "the group_head and type_code fields")
        for tok in ("workflowViewState", "ReviewActionBar", "group_head", "type_code"):
            self.assertIn(tok, kw)

    def test_dotted_call_yields_full_and_tail(self):
        kw = extract_keywords("the handler calls db_docs.create(payload)")
        self.assertIn("db_docs.create", kw)
        self.assertIn("create", kw)

    def test_uppercase_sql_words_and_phrases(self):
        kw = extract_keywords("the SELECT uses WHERE and an ORDER BY with IS NULL")
        for tok in ("SELECT", "WHERE", "ORDER BY", "IS NULL"):
            self.assertIn(tok, kw)

    def test_lowercase_sql_word_is_not_taken(self):
        # "where branches decide" is English, not a query token
        self.assertNotIn("WHERE", extract_keywords("where branches decide flow"))
        self.assertNotIn("where", extract_keywords("where branches decide flow"))

    def test_quoted_literals_kept(self):
        kw = extract_keywords("button bound to 'mark revised' for mode='rejected'")
        self.assertIn("mark revised", kw)
        self.assertIn("rejected", kw)

    def test_titlecase_english_dropped(self):
        kw = extract_keywords("Provide Deliver Search the Component")
        for noise in ("Provide", "Deliver", "Search", "Component"):
            self.assertNotIn(noise, kw)

    def test_upper_snake_axis_id_refs_dropped(self):
        # brief cross-references other axes by UPPER_SNAKE id — not grep seeds
        kw = extract_keywords("git blame the files from FE_RENDER_PATH and "
                              "HEAD_RESOLVER_SQL; keep the type_code field")
        self.assertNotIn("FE_RENDER_PATH", kw)
        self.assertNotIn("HEAD_RESOLVER_SQL", kw)
        self.assertIn("type_code", kw)  # real lowercase snake survives


class TestDocTopics(unittest.TestCase):
    def test_doc_codes_and_phrases(self):
        topics = extract_doc_topics("respect D030 §4 and TR872, the 'head semantics'")
        self.assertIn("D030", topics)
        self.assertIn("TR872", topics)
        self.assertIn("head semantics", topics)


class TestHybridMerge(unittest.TestCase):
    def test_queen_plan_wins_and_local_augments(self):
        task = {
            "id": "HEAD_RESOLVER_SQL",
            "title": "Inspect head-resolution SQL",
            "brief": "Look in server/sql/queries/*.json for the group_head query.",
            "search_plan": {
                "keywords": ["ORDER BY", "IS NULL"],
                "file_globs": ["server/sql/queries/*.json"],
                "doc_topics": ["DB004"],
            },
        }
        sp = task_to_searchplan(task)
        self.assertIsInstance(sp, SearchPlan)
        self.assertEqual(sp.axis_id, "HEAD_RESOLVER_SQL")
        # queen terms present...
        self.assertIn("ORDER BY", sp.keywords)
        self.assertIn("IS NULL", sp.keywords)
        # ...and local augmentation pulled group_head from the brief
        self.assertIn("group_head", sp.keywords)
        self.assertIn("server/sql/queries/*.json", sp.file_globs)
        self.assertIn("DB004", sp.doc_topics)

    def test_queen_terms_come_first(self):
        task = {"id": "X", "brief": "the group_head query",
                "search_plan": {"keywords": ["ORDER BY"]}}
        sp = task_to_searchplan(task)
        self.assertEqual(sp.keywords[0], "ORDER BY")

    def test_absent_queen_plan_falls_back_to_local(self):
        task = {"id": "BE", "title": "endpoints",
                "brief": "inspect server/routers/main.py and db_docs.create()"}
        sp = task_to_searchplan(task)
        self.assertIn("server/routers/main.py", sp.file_globs)
        self.assertIn("db_docs.create", sp.keywords)

    def test_default_globs_when_nothing_named(self):
        task = {"id": "Z", "brief": "find the thing somewhere"}
        sp = task_to_searchplan(task, default_globs=["**/*.py"])
        self.assertEqual(sp.file_globs, ["**/*.py"])

    def test_keyword_cap_respected(self):
        brief = " ".join(f"sym_{i}_x" for i in range(40))
        sp = task_to_searchplan({"id": "C", "brief": brief}, max_keywords=5)
        self.assertLessEqual(len(sp.keywords), 5)

    def test_accepts_axis_id_alias(self):
        sp = task_to_searchplan({"axis_id": "A1", "brief": "x"})
        self.assertEqual(sp.axis_id, "A1")

    def test_malformed_search_plan_ignored(self):
        sp = task_to_searchplan({"id": "M", "brief": "server/x.py", "search_plan": "oops"})
        self.assertIn("server/x.py", sp.file_globs)


class TestVisibilityProbe(unittest.TestCase):
    """N176: a 'not visible / disabled / not rendered' symptom must pull the template
    conditional-render directives into the search, deterministically (not via the queen)."""

    def test_detects_english_visibility_symptoms(self):
        for s in ("the module selector is not visible",
                  "the control doesn't render on screen",
                  "the dropdown is disabled and greyed out",
                  "options no longer appear in the picker"):
            self.assertTrue(is_visibility_symptom(s), s)

    def test_detects_korean_visibility_symptoms(self):
        for s in ("모듈 셀렉터가 화면에 안 보임",
                  "컨트롤이 렌더링되지 않음",
                  "옵션이 노출되지 않습니다",
                  "버튼이 비활성 상태"):
            self.assertTrue(is_visibility_symptom(s), s)

    def test_ignores_unrelated_symptoms(self):
        for s in ("the head doc sorts in the wrong order",
                  "the API returns a 500 on submit",
                  "the total is off by one"):
            self.assertFalse(is_visibility_symptom(s), s)

    def test_probe_appends_directives_after_existing_keywords(self):
        sp = SearchPlan(axis_id="FE", keywords=["moduleSelector", "type_code"],
                        file_globs=["client/**/*.vue"], doc_topics=[])
        out = with_visibility_probe(sp)
        # original terms stay first (queen's precise terms still rank highest)
        self.assertEqual(out.keywords[:2], ["moduleSelector", "type_code"])
        self.assertIn("v-if", out.keywords)
        self.assertIn("v-show", out.keywords)
        self.assertIn("v-for", out.keywords)
        # globs/doc_topics untouched
        self.assertEqual(out.file_globs, ["client/**/*.vue"])

    def test_probe_is_idempotent(self):
        sp = SearchPlan(axis_id="FE", keywords=["v-if"], file_globs=[], doc_topics=[])
        once = with_visibility_probe(sp)
        twice = with_visibility_probe(once)
        # case-insensitive de-dupe: v-if not duplicated
        self.assertEqual(twice.keywords.count("v-if"), 1)
        self.assertEqual(once.keywords, twice.keywords)


class TestCoverageRisk(unittest.TestCase):
    """The queen's coverage_risk self-doubt flag normalises to "thin"|"ok" (B1)."""

    def test_canonical_thin(self):
        self.assertEqual(coverage_risk({"coverage_risk": "thin"}), "thin")

    def test_canonical_ok(self):
        self.assertEqual(coverage_risk({"coverage_risk": "ok"}), "ok")

    def test_synonyms_coerce_to_thin(self):
        for v in ("high", "risky", "low", "weak", "uncertain", "true", "THIN", " Thin "):
            self.assertEqual(coverage_risk({"coverage_risk": v}), "thin", v)

    def test_bool_true_is_thin_false_is_ok(self):
        self.assertEqual(coverage_risk({"coverage_risk": True}), "thin")
        self.assertEqual(coverage_risk({"coverage_risk": False}), "ok")

    def test_absent_or_empty_defaults_ok(self):
        # Opt-in: a queen that never emitted the field is NOT treated as flagged.
        for task in ({}, {"coverage_risk": ""}, {"coverage_risk": None},
                     {"coverage_risk": "anything-else"}, {"id": "A"}):
            self.assertEqual(coverage_risk(task), "ok", task)


if __name__ == "__main__":
    unittest.main()
